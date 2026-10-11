"""
R03b_lock_hub_genes.py — fix the final hub-gene set used by every later revision step.

Rule (decided before downstream analysis): a gene is a hub gene if it is in the consensus
centrality top 10 in >= 6 of the 9 STRING networks (3 evidence types x 3 cut-offs) from R03.

Output: analysis/hub_genes_final.csv   (single source of truth for all later scripts)
"""
import pandas as pd
from pathlib import Path

NEW = Path(__file__).resolve().parents[1]
R = pd.read_csv(NEW / "03_PPI_hub_selection" / "consensus_rank_across_9_networks.csv", index_col=0)
G = pd.read_csv(NEW / "02_WGCNA_signed" / "A_key_module_ME2_all_genes.csv").set_index("gene_name")
prim = pd.read_csv(NEW / "03_PPI_hub_selection" / "centrality_E2_0.7.csv", index_col=0)

MIN_NETWORKS = 6
hub = R[R.n_top10 >= MIN_NETWORKS][["n_top10", "median_rank"]].copy()
hub["primary_network_rank_E2_0.7"] = prim.reindex(hub.index)["consensus_rank"]
hub = hub.join(G[["gene_id", "MM_own", "GS_tumour", "log2FoldChange", "padj"]])
hub = hub.rename(columns={"n_top10": "networks_in_top10_of_9", "MM_own": "MM_ME2_signed",
                          "GS_tumour": "GS_tumour_signed"})
hub = hub.sort_values(["networks_in_top10_of_9", "median_rank"], ascending=[False, True])
hub.index.name = "gene"
hub.to_csv(NEW / "hub_genes_final.csv")
print(f"Locked {len(hub)} hub genes (top-10 in >= {MIN_NETWORKS}/9 networks):")
print(hub.round(3).to_string())
