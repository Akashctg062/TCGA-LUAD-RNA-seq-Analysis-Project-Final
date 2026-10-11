"""
R03_ppi_hub_selection.py  —  Revision step 3 (PI comment M3)
=============================================================
Hub-gene selection on the FULL key-module candidate set (instead of a 9-node network).

Input : analysis/02_WGCNA_signed/A_key_module_ME2_hub_candidates_MM0.8_GS0.5_DEGup.csv  (86 genes)
STRING: v12.0, Homo sapiens (9606). Downloaded once and cached in analysis/03_PPI_hub_selection/string_cache/
        (needs internet the first time; afterwards runs offline from the cache)

Robustness design  (3 evidence definitions x 3 confidence cut-offs = 9 networks)
  evidence  E1  STRING functional network, combined score (all channels)
            E2  functional network, combined score RECOMPUTED WITHOUT the co-expression channel
                (removes circularity with WGCNA, which is itself co-expression based)
            E3  STRING physical-interaction network (physical subnetwork score)
  cut-offs  0.4 (medium), 0.7 (high), 0.9 (highest)

Centralities per network (cytoHubba-style): Degree, Betweenness, Closeness, MCC, MNC
Consensus rank per network = mean of the five per-measure ranks.
Robustness = number of the 9 networks in which a gene is in the top 10 by consensus rank.

Outputs: analysis/03_PPI_hub_selection/
"""
import json, math, time
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NEW = ROOT / "analysis"
OUT = NEW / "03_PPI_hub_selection"; OUT.mkdir(parents=True, exist_ok=True)
CACHE = OUT / "string_cache"; CACHE.mkdir(exist_ok=True)
CAND = NEW / "02_WGCNA_signed" / "A_key_module_ME2_hub_candidates_MM0.8_GS0.5_DEGup.csv"
API = "https://version-12-0.string-db.org/api"
HUBS = ["BIRC5", "CCNB1", "CCNB2", "CDC20", "KIF2C"]
CUTS = [0.4, 0.7, 0.9]
TOP = 10
PRIOR = 0.041      # STRING prior used in its score combination

report = []
def log(s=""):
    print(s, flush=True); report.append(str(s))

cand = pd.read_csv(CAND)
genes = cand.gene_name.tolist()
log("=" * 70); log("R03  PPI HUB-GENE SELECTION (PI comment M3)"); log("=" * 70)
log(f"Candidates from signed-WGCNA key module ME2 (MM>0.8, GS>0.5, up-DEG): {len(genes)}")

# ── 1. STRING download (cached) ──
def fetch(network_type):
    f = CACHE / f"string_v12_{network_type}_score150.tsv"
    if f.exists():
        return pd.read_csv(f, sep="\t")
    import requests
    r = requests.post(f"{API}/tsv/network", timeout=120, data={
        "identifiers": "\r".join(genes), "species": 9606, "required_score": 150,
        "network_type": network_type, "caller_identity": "tcga_luad_revision"})
    r.raise_for_status()
    f.write_text(r.text, encoding="utf-8")
    time.sleep(1)
    return pd.read_csv(f, sep="\t")

def fetch_ids():
    f = CACHE / "string_v12_id_mapping.tsv"
    if f.exists():
        return pd.read_csv(f, sep="\t")
    import requests
    r = requests.post(f"{API}/tsv/get_string_ids", timeout=120, data={
        "identifiers": "\r".join(genes), "species": 9606, "echo_query": 1, "limit": 1,
        "caller_identity": "tcga_luad_revision"})
    r.raise_for_status()
    f.write_text(r.text, encoding="utf-8")
    return pd.read_csv(f, sep="\t")

ids = fetch_ids()
func = fetch("functional")
phys = fetch("physical")
mapped = set(ids.queryItem)
log(f"Mapped to STRING v12: {len(mapped)}/{len(genes)}; unmapped: {sorted(set(genes) - mapped)}")
renamed = ids[ids.queryItem != ids.preferredName][["queryItem", "preferredName"]]
if len(renamed): log(f"Renamed by STRING: {renamed.values.tolist()}")
back = dict(zip(ids.preferredName, ids.queryItem))      # use our gene symbols throughout

# ── 2. Evidence definitions ──
def combine(df, channels):
    """STRING score combination with prior correction (von Mering et al. 2005)."""
    p = 1.0
    for c in channels:
        s = ((df[c] - PRIOR) / (1 - PRIOR)).clip(lower=0)
        p = p * (1 - s)
    s = 1 - p
    return s * (1 - PRIOR) + PRIOR

func["E1"] = func["score"]
func["E2"] = combine(func, ["nscore", "fscore", "pscore", "escore", "dscore", "tscore"])  # no ascore
phys["E3"] = phys["score"]
evid = {"E1_functional_all_channels": (func, "E1"),
        "E2_functional_no_coexpression": (func, "E2"),
        "E3_physical": (phys, "E3")}

def build(df, col, cut):
    G = nx.Graph()
    G.add_nodes_from([back.get(g, g) for g in ids.preferredName])
    d = df[df[col] >= cut]
    for a, b, w in zip(d.preferredName_A, d.preferredName_B, d[col]):
        G.add_edge(back.get(a, a), back.get(b, b), weight=float(w))
    return G

# ── 3. cytoHubba-style centralities ──
def mcc(G):
    """Maximal Clique Centrality (Chin et al. 2014): sum over maximal cliques C containing v of (|C|-1)!;
    nodes whose neighbours are not connected get MCC = degree."""
    score = {v: 0 for v in G}
    for C in nx.find_cliques(G):
        if len(C) < 2: continue
        f = math.factorial(len(C) - 1)
        for v in C: score[v] += f
    for v in G:
        nb = list(G[v])
        if len(nb) and G.subgraph(nb).number_of_edges() == 0:
            score[v] = len(nb)
    return score

def mnc(G):
    out = {}
    for v in G:
        sub = G.subgraph(G[v])
        out[v] = max((len(c) for c in nx.connected_components(sub)), default=0)
    return out

def centralities(G):
    t = pd.DataFrame({
        "Degree": dict(G.degree()),
        "Betweenness": nx.betweenness_centrality(G),
        "Closeness": nx.closeness_centrality(G),
        "MCC": {k: float(v) for k, v in mcc(G).items()},
        "MNC": mnc(G),
    })
    for c in ["Degree", "Betweenness", "Closeness", "MCC", "MNC"]:
        t[f"rank_{c}"] = t[c].rank(ascending=False, method="average")
    t["consensus_rank_score"] = t.filter(like="rank_").mean(1)
    t["consensus_rank"] = t.consensus_rank_score.rank(method="min")
    return t.sort_values("consensus_rank")

# ── 4. Run 9 networks ──
summary, ranks, tops = [], {}, {}
for ename, (df, col) in evid.items():
    for cut in CUTS:
        key = f"{ename.split('_')[0]}_{cut}"
        G = build(df, col, cut)
        Gc = G.subgraph([n for n in G if G.degree(n) > 0]).copy()
        t = centralities(Gc) if Gc.number_of_edges() else pd.DataFrame()
        t.to_csv(OUT / f"centrality_{key}.csv")
        comps = sorted((len(c) for c in nx.connected_components(Gc)), reverse=True) if len(Gc) else []
        summary.append(dict(network=key, evidence=ename, cutoff=cut, nodes_with_edges=Gc.number_of_nodes(),
                            edges=Gc.number_of_edges(), density=nx.density(Gc) if len(Gc) > 1 else 0,
                            largest_component=comps[0] if comps else 0,
                            top10=", ".join(t.index[:TOP]) if len(t) else ""))
        ranks[key] = t["consensus_rank"] if len(t) else pd.Series(dtype=float)
        tops[key] = list(t.index[:TOP]) if len(t) else []
summ = pd.DataFrame(summary)
summ.to_csv(OUT / "network_summary_9_settings.csv", index=False)
log("\nNetworks (3 evidence types x 3 cut-offs):")
log(summ[["network", "nodes_with_edges", "edges", "density", "largest_component"]]
    .to_string(index=False, float_format=lambda v: f"{v:.3f}"))

R = pd.DataFrame(ranks).reindex(genes)
R["n_top10"] = sum(R.index.isin(tops[k]) for k in tops)
R["n_networks_present"] = R.drop(columns="n_top10").notna().sum(1)
R["median_rank"] = R.drop(columns=["n_top10", "n_networks_present"]).median(1)
R = R.sort_values(["n_top10", "median_rank"], ascending=[False, True])
R.to_csv(OUT / "consensus_rank_across_9_networks.csv")

log(f"\nGenes most often in the consensus top {TOP} across the 9 networks:")
log(R[["n_top10", "median_rank", "n_networks_present"]].head(20).to_string(float_format=lambda v: f"{v:.1f}"))
log("\nOriginal five hub genes:")
log(R.loc[HUBS, ["n_top10", "median_rank", "n_networks_present"]].to_string(float_format=lambda v: f"{v:.1f}"))

# primary network = E2 (no co-expression) at 0.7, the least circular high-confidence definition
prim = pd.read_csv(OUT / "centrality_E2_0.7.csv", index_col=0)
log("\nPrimary network E2 (no co-expression channel), score >= 0.7 — consensus top 15:")
log(prim.head(15)[["Degree", "Betweenness", "Closeness", "MCC", "MNC", "consensus_rank"]]
    .to_string(float_format=lambda v: f"{v:.3g}"))

# ── 5. Figures ──
hm = R.drop(columns=["n_top10", "n_networks_present", "median_rank"]).head(25)
fig, ax = plt.subplots(figsize=(8.5, 9))
sns.heatmap(hm, annot=True, fmt=".0f", cmap="viridis_r", vmin=1, vmax=40, ax=ax,
            cbar_kws={"label": "Consensus rank (1 = most central)"})
for lab in ax.get_yticklabels():
    if lab.get_text() in HUBS: lab.set_color("#c0392b"); lab.set_fontweight("bold")
ax.set_xticklabels([c.replace("_", "\n≥") for c in hm.columns], rotation=0, fontsize=8)
ax.set_title("Hub-gene robustness: consensus centrality rank\n(E1 all channels, E2 no co-expression, E3 physical)")
plt.tight_layout(); plt.savefig(OUT / "Fig_consensus_rank_heatmap.png", dpi=300); plt.close()

G = build(func, "E2", 0.7)
G = G.subgraph([n for n in G if G.degree(n) > 0]).copy()
top15 = list(prim.index[:15])
pos = nx.spring_layout(G, seed=42, k=0.5)
fig, ax = plt.subplots(figsize=(10, 9))
nx.draw_networkx_edges(G, pos, alpha=.12, width=.6, ax=ax)
col = ["#c0392b" if n in HUBS else ("#e67e22" if n in top15 else "#95a5a6") for n in G]
size = [40 + 6 * G.degree(n) for n in G]
nx.draw_networkx_nodes(G, pos, node_color=col, node_size=size, linewidths=.5, edgecolors="white", ax=ax)
nx.draw_networkx_labels(G, pos, labels={n: n for n in top15 + HUBS}, font_size=7, ax=ax)
ax.set_title(f"STRING v12 network of ME2 candidates (no co-expression evidence, score ≥ 0.7)\n"
             f"{G.number_of_nodes()} nodes, {G.number_of_edges()} edges; red = five hub genes, orange = other top-15")
ax.axis("off"); plt.tight_layout(); plt.savefig(OUT / "Fig_PPI_network_E2_0.7.png", dpi=300); plt.close()

(OUT / "R03_parameters.json").write_text(json.dumps(dict(
    string_version="12.0", species=9606, download_min_score=0.15, evidence=list(evid), cutoffs=CUTS,
    centralities=["Degree", "Betweenness", "Closeness", "MCC", "MNC"], consensus="mean of per-measure ranks",
    robustness=f"count of 9 networks with gene in consensus top {TOP}",
    primary_network="E2 (combined score without co-expression channel) >= 0.7"), indent=2))
(OUT / "R03_report.txt").write_text("\n".join(report), encoding="utf-8")
log(f"\nOutputs in {OUT}")
