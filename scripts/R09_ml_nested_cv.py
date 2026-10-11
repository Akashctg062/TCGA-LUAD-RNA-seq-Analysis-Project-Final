"""
R09_ml_nested_cv.py — Revision step 9 (PI comment M4; TRIPOD+AI-style reporting)
=================================================================================
Part 0  Reproduce the two errors of the original analysis (for the response letter)
        * MLP AUC 0.372: early_stopping=True holds out 10% of each training fold (~5 normal samples),
          training stopped after 12-30 iterations -> under-fitted network
        * specificity 76.3% (text) vs 79.7% (Fig. S14C): text = 5-fold CV predictions,
          figure = model fitted and evaluated on the same samples (apparent performance)
Part A  Diagnostic (tumour vs normal), TCGA-LUAD 517 T / 59 N
        Repeated nested CV: outer 5-fold x 5 repeats; inner 3-fold GridSearch (AUC)
        Pipelines: StandardScaler + {LR, SVM-RBF, RF, XGBoost, MLP}, class-weighted
        Feature sets: HUB7 (fixed, selected on all TCGA data = optimistic)
                      FOLD7 (7 most up-regulated genes re-selected INSIDE each training fold by Welch t;
                             honest estimate of "select DEGs + classify")
        Threshold: Youden index on inner out-of-fold predictions of the training fold, then applied to test fold
        Metrics: AUC, average precision, balanced accuracy, MCC, sensitivity, specificity (+ at 0.5)
        External: models refitted on all TCGA (HUB7, within-cohort z-scores), locked threshold,
                  tested on GSE31210 226 T / 20 N; bootstrap 95% CI
Part B  Prognostic (the clinically relevant task): overall survival
        Models: Cox clinical (age, sex, stage) | Cox clinical + HUB7 mean score |
                elastic-net Cox clinical + 7 genes | random survival forest clinical + 7 genes
        TCGA repeated CV (5-fold x 5) Harrell C; locked models applied to GSE31210, GSE68465, GSE72094
        (C-index with bootstrap CI; calibration slope of the linear predictor for Cox models)
SHAP is not reported: the seven genes are highly collinear (r ~ 0.85-0.95), so per-gene attributions are unstable.
Outputs: analysis/09_ML/
"""
import warnings; warnings.filterwarnings("ignore")
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import ttest_ind
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import (StratifiedKFold, RepeatedStratifiedKFold, GridSearchCV,
                                     cross_val_predict, KFold)
from sklearn.metrics import (roc_auc_score, average_precision_score, balanced_accuracy_score,
                             matthews_corrcoef, confusion_matrix, roc_curve)
from xgboost import XGBClassifier
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv

ROOT = Path(__file__).resolve().parents[2]
NEW = ROOT / "analysis"
OUT = NEW / "09_ML"; OUT.mkdir(exist_ok=True)
CK = OUT / "checkpoints"; CK.mkdir(exist_ok=True)   # resumable: finished models are cached here
SEED = 42
rng = np.random.default_rng(SEED)
HUB7 = pd.read_csv(NEW / "hub_genes_final.csv").gene.tolist()
OLD5 = ["BIRC5", "CCNB1", "CCNB2", "CDC20", "KIF2C"]
fmt = lambda v: f"{v:.3g}"
t0 = time.time()
def log(*a):
    print(f"[{time.time()-t0:6.0f}s]", *a, flush=True)

# ── data ──
counts = pd.read_csv(ROOT / "data" / "filtered_counts.csv", index_col=0)
gmap = pd.read_csv(ROOT / "data" / "gene_id_mapping.csv", index_col=0)["gene_name"]
meta = pd.read_csv(ROOT / "data" / "metadata.csv").set_index("sample_submitter_id")
lc = np.log2(counts.div(counts.sum(0), axis=1) * 1e6 + 1)
lc.index = gmap.reindex(lc.index).fillna(pd.Series(lc.index, index=lc.index)).values
lc = lc[~lc.index.duplicated()]
lc = lc[lc.mean(1) > 1]                                     # expressed genes for in-fold selection
E = lc.T
y = (meta.loc[E.index, "condition"] == "Tumor").astype(int).values
log(f"TCGA: {E.shape[0]} samples ({y.sum()} T / {(1-y).sum()} N), {E.shape[1]} expressed genes")

# ══════════ Part 0: reproduce original errors ══════════
Xo = StandardScaler().fit_transform(E[OLD5].values)
cv0 = StratifiedKFold(5, shuffle=True, random_state=42)
p_old = cross_val_predict(MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42,
                                        early_stopping=True), Xo, y, cv=cv0, method="predict_proba")[:, 1]
iters = [MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42, early_stopping=True)
         .fit(Xo[tr], y[tr]).n_iter_ for tr, _ in cv0.split(Xo, y)]
p_fix = cross_val_predict(MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=2000, random_state=42),
                          Xo, y, cv=cv0, method="predict_proba")[:, 1]
p_lr = cross_val_predict(LogisticRegression(max_iter=1000, random_state=42), Xo, y, cv=cv0, method="predict_proba")[:, 1]
p_app = LogisticRegression(max_iter=1000, random_state=42).fit(Xo, y).predict_proba(Xo)[:, 1]
def cm_metrics(yt, pred):
    tn, fp, fn, tp = confusion_matrix(yt, pred).ravel()
    return dict(TN=tn, FP=fp, FN=fn, TP=tp, sensitivity=tp / (tp + fn), specificity=tn / (tn + fp),
                accuracy=(tp + tn) / len(yt), PPV=tp / (tp + fp))
diag0 = pd.DataFrame([
    dict(item="original MLP (early stopping)", AUC=roc_auc_score(y, p_old), iterations=str(iters),
         accuracy=np.mean((p_old >= .5) == y), frac_predicted_tumour=np.mean(p_old >= .5)),
    dict(item="same MLP without early stopping", AUC=roc_auc_score(y, p_fix)),
    dict(item="LR 5-fold CV predictions (source of manuscript text)", **cm_metrics(y, p_lr >= .5)),
    dict(item="LR apparent fit on all data (source of Fig. S14C)", **cm_metrics(y, p_app >= .5)),
])
diag0.to_csv(OUT / "part0_reproduction_of_original_errors.csv", index=False)
log("Part 0 done")

# ══════════ Part A: diagnostic nested CV ══════════
class TopUpGenes(BaseEstimator, TransformerMixin):
    """Select the k most up-regulated genes (Welch t, tumour vs normal) using training data only."""
    def __init__(self, k=7): self.k = k
    def fit(self, X, y):
        t, _ = ttest_ind(X[y == 1], X[y == 0], equal_var=False)
        self.idx_ = np.argsort(-np.nan_to_num(t))[: self.k]; return self
    def transform(self, X): return X[:, self.idx_]

pos_w = (y == 0).sum() / (y == 1).sum()
MODELS = {
    "Logistic regression": (LogisticRegression(max_iter=5000, class_weight="balanced"), {"m__C": [0.01, 0.1, 1, 10]}),
    "SVM (RBF)": (SVC(probability=True, class_weight="balanced", random_state=SEED), {"m__C": [0.1, 1, 10]}),
    "Random forest": (RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=SEED, n_jobs=2),
                      {"m__max_depth": [3, None]}),
    "XGBoost": (XGBClassifier(n_estimators=200, learning_rate=0.1, scale_pos_weight=pos_w, eval_metric="logloss",
                              random_state=SEED, verbosity=0, n_jobs=2), {"m__max_depth": [2, 4]}),
    "MLP": (MLPClassifier(hidden_layer_sizes=(16,), max_iter=3000, random_state=SEED), {"m__alpha": [1e-4, 1e-2]}),
}

def fold_metrics(yt, prob, thr):
    pred = (prob >= thr).astype(int); pred05 = (prob >= .5).astype(int)
    tn, fp, fn, tp = confusion_matrix(yt, pred, labels=[0, 1]).ravel()
    return dict(AUC=roc_auc_score(yt, prob), AP=average_precision_score(yt, prob),
                balanced_accuracy=balanced_accuracy_score(yt, pred), MCC=matthews_corrcoef(yt, pred),
                sensitivity=tp / (tp + fn), specificity=tn / (tn + fp),
                balanced_accuracy_0_5=balanced_accuracy_score(yt, pred05), MCC_0_5=matthews_corrcoef(yt, pred05))

def youden(yt, prob):
    f, t, th = roc_curve(yt, prob); return th[np.argmax(t - f)]

outer = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=SEED)
inner = StratifiedKFold(3, shuffle=True, random_state=SEED)
Xall = E.values; hub_idx = [list(E.columns).index(g) for g in HUB7]
rows, oof = [], {}
for fs in ["HUB7", "FOLD7"]:
    for name, (est, grid) in MODELS.items():
        steps = ([("sel", TopUpGenes(7))] if fs == "FOLD7" else []) + [("sc", StandardScaler()), ("m", clone(est))]
        pipe = Pipeline(steps)
        X = Xall if fs == "FOLD7" else Xall[:, hub_idx]
        ck = CK / f"A_{fs}_{name.replace(' ', '_').replace('(', '').replace(')', '')}.pkl"
        if ck.exists():
            r_, o_ = pd.read_pickle(ck); rows.extend(r_); oof[(fs, name)] = o_; log(f"A {fs:5s} {name} (cached)"); continue
        n_before = len(rows)
        oof_rep0 = np.full(len(y), np.nan)
        for k, (tr, te) in enumerate(outer.split(X, y)):
            gs = GridSearchCV(pipe, grid, cv=inner, scoring="roc_auc", n_jobs=1).fit(X[tr], y[tr])
            thr = youden(y[tr], cross_val_predict(gs.best_estimator_, X[tr], y[tr], cv=inner, method="predict_proba")[:, 1])
            p = gs.best_estimator_.predict_proba(X[te])[:, 1]
            rows.append(dict(features=fs, model=name, fold=k, threshold=thr, best=str(gs.best_params_), **fold_metrics(y[te], p, thr)))
            if k < 5: oof_rep0[te] = p
        oof[(fs, name)] = oof_rep0
        pd.to_pickle((rows[n_before:], oof_rep0), ck)
        log(f"A {fs:5s} {name} done")
cvres = pd.DataFrame(rows); cvres.to_csv(OUT / "A_diagnostic_nested_cv_per_fold.csv", index=False)
MET = ["AUC", "AP", "balanced_accuracy", "MCC", "sensitivity", "specificity", "balanced_accuracy_0_5", "MCC_0_5"]
summ = cvres.groupby(["features", "model"])[MET].agg(["mean", lambda s: s.quantile(.025), lambda s: s.quantile(.975)])
summ.columns = [f"{m}_{s}" for m, s in zip(summ.columns.get_level_values(0),
                                          ["mean", "lo", "hi"] * len(MET))]
summ.to_csv(OUT / "A_diagnostic_nested_cv_summary.csv")
# pooled confusion matrix (repeat 1, Youden threshold from all folds' median) — single consistent source
cms = []
for (fs, name), p in oof.items():
    thr = cvres[(cvres.features == fs) & (cvres.model == name)].threshold.iloc[:5].median()
    cms.append(dict(features=fs, model=name, threshold=thr, **cm_metrics(y, p >= thr),
                    MCC=matthews_corrcoef(y, p >= thr), balanced_accuracy=balanced_accuracy_score(y, p >= thr)))
pd.DataFrame(cms).to_csv(OUT / "A_pooled_out_of_fold_confusion_matrices.csv", index=False)

# unfitted baselines
Zall = (E - E.mean()) / E.std()
base = pd.DataFrame([dict(score="HUB7 mean z (no fitting)", AUC=roc_auc_score(y, Zall[HUB7].mean(1))),
                     dict(score="MKI67 alone", AUC=roc_auc_score(y, E["MKI67"]))])
base.to_csv(OUT / "A_unfitted_baselines.csv", index=False)

# external diagnostic test: GSE31210
DAT = NEW / "07_external_validation" / "data"
Eg = pd.read_csv(DAT / "GSE31210_expression_genes.csv", index_col=0)
Cg = pd.read_csv(DAT / "GSE31210_clinical.csv", index_col=0)
Xg = ((Eg[HUB7] - Eg[HUB7].mean()) / Eg[HUB7].std()).loc[Cg.index].values; yg = Cg.tumour.values
Xt = Zall[HUB7].values
ext = []; ext_prob = {}
for name, (est, grid) in MODELS.items():
    gs = GridSearchCV(Pipeline([("sc", StandardScaler()), ("m", clone(est))]), grid, cv=inner, scoring="roc_auc").fit(Xt, y)
    thr = youden(y, cross_val_predict(gs.best_estimator_, Xt, y, cv=inner, method="predict_proba")[:, 1])
    p = gs.best_estimator_.predict_proba(Xg)[:, 1]; ext_prob[name] = p
    bt = []
    for _ in range(1000):
        i = rng.integers(0, len(yg), len(yg))
        if len(set(yg[i])) == 2: bt.append([roc_auc_score(yg[i], p[i]), balanced_accuracy_score(yg[i], p[i] >= thr),
                                            matthews_corrcoef(yg[i], p[i] >= thr)])
    bt = np.array(bt)
    ext.append(dict(model=name, n_tumour=int(yg.sum()), n_normal=int((1 - yg).sum()), locked_threshold=thr,
                    **{k: v for k, v in fold_metrics(yg, p, thr).items()},
                    AUC_lo=np.percentile(bt[:, 0], 2.5), AUC_hi=np.percentile(bt[:, 0], 97.5),
                    BA_lo=np.percentile(bt[:, 1], 2.5), BA_hi=np.percentile(bt[:, 1], 97.5),
                    MCC_lo=np.percentile(bt[:, 2], 2.5), MCC_hi=np.percentile(bt[:, 2], 97.5)))
ext = pd.DataFrame(ext); ext.to_csv(OUT / "A_external_GSE31210_diagnostic.csv", index=False)
log("Part A done")

# ══════════ Part B: prognostic ══════════
clin = pd.read_csv(NEW / "04_clinical" / "clinical_tumour_harmonised.csv").set_index("sample_submitter_id")
def clin_frame(C, Ex, os_days="OS_days", os_ev="OS_event", stage="stage", male=None):
    d = pd.DataFrame(index=C.index)
    d["T"] = C[os_days]; d["E"] = C[os_ev]; d["age10"] = C["age"] / 10
    d["male"] = male if male is not None else C["male"]
    for s in ["II", "III", "IV"]: d[f"st_{s}"] = (C[stage] == s).astype(float).where(C[stage].notna())
    Z = (Ex[HUB7] - Ex[HUB7].mean()) / Ex[HUB7].std()
    d = d.join(Z.add_prefix("g_"))
    d["score"] = Z.mean(1); d["score"] = (d.score - d.score.mean()) / d.score.std()
    return d.dropna()
Etum = E.loc[clin.index]
dT = clin_frame(clin, Etum, male=(clin.sex == "Male").astype(float))
CLIN = ["age10", "male", "st_II", "st_III", "st_IV"]; GENES = [f"g_{g}" for g in HUB7]
VAL = {}
for gse in ["GSE31210", "GSE68465", "GSE72094"]:
    C = pd.read_csv(DAT / f"{gse}_clinical.csv", index_col=0); C = C[C.tumour == 1]
    Ex = pd.read_csv(DAT / f"{gse}_expression_genes.csv", index_col=0).loc[C.index]
    d = clin_frame(C, Ex)
    for s in ["II", "III", "IV"]:
        if f"st_{s}" not in d: d[f"st_{s}"] = 0.0
    VAL[gse] = d

def fit_model(kind, d, pen=None):
    if kind == "Cox clinical":
        return CoxPHFitter().fit(d[["T", "E"] + CLIN], "T", "E")
    if kind == "Cox clinical + HUB7 score":
        return CoxPHFitter().fit(d[["T", "E"] + CLIN + ["score"]], "T", "E")
    if kind == "Elastic-net Cox clinical + 7 genes":
        if pen is None:   # inner CV for penalizer
            best, bc = 0.1, -1
            for pz in [0.01, 0.05, 0.1, 0.3]:
                cs = []
                for tr, te in KFold(3, shuffle=True, random_state=SEED).split(d):
                    m = CoxPHFitter(penalizer=pz, l1_ratio=0.5).fit(d.iloc[tr][["T", "E"] + CLIN + GENES], "T", "E")
                    cs.append(concordance_index(d.iloc[te]["T"], -m.predict_partial_hazard(d.iloc[te]), d.iloc[te]["E"]))
                if np.mean(cs) > bc: best, bc = pz, np.mean(cs)
            pen = best
        m = CoxPHFitter(penalizer=pen, l1_ratio=0.5).fit(d[["T", "E"] + CLIN + GENES], "T", "E"); m.pen_ = pen; return m
    if kind == "Random survival forest clinical + 7 genes":
        return RandomSurvivalForest(n_estimators=300, min_samples_leaf=15, max_features="sqrt", random_state=SEED,
                                    n_jobs=-1).fit(d[CLIN + GENES].values, Surv.from_arrays(d.E.astype(bool), d["T"]))
def risk(m, d):
    if isinstance(m, RandomSurvivalForest): return m.predict(d[CLIN + GENES].values)
    return np.log(m.predict_partial_hazard(d).values)
KINDS = ["Cox clinical", "Cox clinical + HUB7 score", "Elastic-net Cox clinical + 7 genes",
         "Random survival forest clinical + 7 genes"]

prow = []
for rep in range(5):
    ck = CK / f"B_rep{rep}.pkl"
    if ck.exists():
        prow.extend(pd.read_pickle(ck)); log(f"B repeat {rep} (cached)"); continue
    n_before = len(prow)
    for k, (tr, te) in enumerate(KFold(5, shuffle=True, random_state=SEED + rep).split(dT)):
        a, b = dT.iloc[tr], dT.iloc[te]
        for kind in KINDS:
            m = fit_model(kind, a)
            prow.append(dict(rep=rep, fold=k, model=kind, C=concordance_index(b["T"], -risk(m, b), b["E"])))
    pd.to_pickle(prow[n_before:], ck)
    log(f"B repeat {rep} done")
pcv = pd.DataFrame(prow); pcv.to_csv(OUT / "B_prognostic_cv_per_fold.csv", index=False)
wide = pcv.pivot_table(index=["rep", "fold"], columns="model", values="C")
pcs = pd.DataFrame({k: dict(C_mean=wide[k].mean(), C_lo=wide[k].quantile(.025), C_hi=wide[k].quantile(.975),
                            delta_vs_clinical=(wide[k] - wide["Cox clinical"]).mean(),
                            delta_lo=(wide[k] - wide["Cox clinical"]).quantile(.025),
                            delta_hi=(wide[k] - wide["Cox clinical"]).quantile(.975)) for k in KINDS}).T
pcs.to_csv(OUT / "B_prognostic_cv_summary.csv")

final = {k: fit_model(k, dT) for k in KINDS}
erow = []
for gse, d in VAL.items():
    for kind, m in final.items():
        r = risk(m, d); c = concordance_index(d["T"], -r, d["E"])
        bt = []
        for _ in range(500):
            i = rng.integers(0, len(d), len(d)); bt.append(concordance_index(d["T"].values[i], -r[i], d["E"].values[i]))
        slope = np.nan
        if not isinstance(m, RandomSurvivalForest):
            lp = pd.DataFrame({"T": d["T"], "E": d["E"], "lp": r - r.mean()})
            slope = CoxPHFitter().fit(lp, "T", "E").params_["lp"]
        erow.append(dict(cohort=gse, model=kind, n=len(d), events=int(d.E.sum()), C=c,
                         C_lo=np.percentile(bt, 2.5), C_hi=np.percentile(bt, 97.5), calibration_slope=slope))
pext = pd.DataFrame(erow); pext.to_csv(OUT / "B_prognostic_external_locked_models.csv", index=False)
pd.Series(final["Elastic-net Cox clinical + 7 genes"].params_, name="coef").to_csv(OUT / "B_elastic_net_cox_coefficients.csv")
log("Part B done")

# ══════════ figures ══════════
fig, ax = plt.subplots(1, 3, figsize=(18, 5.2))
order = list(MODELS)
for j, (met, lab) in enumerate([("AUC", "AUC"), ("balanced_accuracy", "Balanced accuracy (Youden)"), ("MCC", "MCC (Youden)")]):
    for i, fs in enumerate(["HUB7", "FOLD7"]):
        s = summ.loc[fs].loc[order]
        x = np.arange(len(order)) + (i - .5) * .38
        ax[j].bar(x, s[f"{met}_mean"], .36, yerr=[s[f"{met}_mean"] - s[f"{met}_lo"], s[f"{met}_hi"] - s[f"{met}_mean"]],
                  capsize=3, color=["#c0392b", "#7f8c8d"][i], label=["HUB7 (fixed genes)", "7 genes re-selected in each fold"][i])
    e = ext.set_index("model").loc[order]
    ax[j].scatter(np.arange(len(order)), e[{"AUC": "AUC", "balanced_accuracy": "balanced_accuracy", "MCC": "MCC"}[met]],
                  marker="D", color="#2980b9", zorder=3, label="External GSE31210 (HUB7, locked)")
    ax[j].set_xticks(range(len(order))); ax[j].set_xticklabels(order, rotation=20, ha="right")
    ax[j].set_ylim(0.5 if met != "MCC" else 0, 1.0); ax[j].set_title(lab)
    ax[j].spines[["top", "right"]].set_visible(False)
ax[0].legend(fontsize=8, frameon=False, loc="lower left")
fig.suptitle("Diagnostic classification (tumour vs normal): repeated nested CV in TCGA-LUAD (mean, 95% range over 25 outer folds)")
plt.tight_layout(); plt.savefig(OUT / "Fig_A_diagnostic_nested_cv.png", dpi=300); plt.close()

fig, ax = plt.subplots(figsize=(11, 5))
cohorts = ["TCGA (CV)"] + list(VAL)
cols = ["#7f8c8d", "#c0392b", "#e67e22", "#8e44ad"]
for i, kind in enumerate(KINDS):
    vals, lo, hi = [pcs.loc[kind, "C_mean"]], [pcs.loc[kind, "C_lo"]], [pcs.loc[kind, "C_hi"]]
    for gse in VAL:
        r = pext[(pext.cohort == gse) & (pext.model == kind)].iloc[0]; vals.append(r.C); lo.append(r.C_lo); hi.append(r.C_hi)
    x = np.arange(len(cohorts)) + (i - 1.5) * .2
    ax.errorbar(x, vals, yerr=[np.array(vals) - lo, np.array(hi) - vals], fmt="o", color=cols[i], capsize=3, label=kind)
ax.set_xticks(range(len(cohorts))); ax.set_xticklabels(cohorts); ax.set_ylabel("Harrell C-index (OS)")
ax.axhline(.5, ls=":", c="grey"); ax.legend(fontsize=8, frameon=False)
ax.set_title("Prognostic models: TCGA repeated CV and locked external validation")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.savefig(OUT / "Fig_B_prognostic_models.png", dpi=300); plt.close()

rep = ["R09 ML (nested CV, external validation)", "=" * 60,
       "\nPart 0 — reproduction of the original errors:", diag0.to_string(index=False, float_format=fmt),
       "\nPart A — diagnostic nested CV (mean [2.5-97.5% over 25 outer folds]):",
       summ[[c for c in summ.columns if c.split("_")[-1] == "mean" or True]].round(3).to_string(),
       "\nPooled out-of-fold confusion matrices (repeat 1):", pd.DataFrame(cms).to_string(index=False, float_format=fmt),
       "\nUnfitted baselines:", base.to_string(index=False, float_format=fmt),
       "\nExternal GSE31210 (locked models, locked threshold):", ext.to_string(index=False, float_format=fmt),
       "\nPart B — prognostic, TCGA repeated CV:", pcs.to_string(float_format=fmt),
       "\nPart B — locked models in GEO:", pext.to_string(index=False, float_format=fmt),
       f"\nElastic-net penalizer chosen on full TCGA: {final['Elastic-net Cox clinical + 7 genes'].pen_}"]
(OUT / "R09_report.txt").write_text("\n".join(rep), encoding="utf-8")
log("all done")
