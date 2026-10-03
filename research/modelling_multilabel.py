import re
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
from sklearn.metrics import average_precision_score

feature = pd.read_csv("features.csv", index_col=0)
labels = pd.read_csv("labels.csv", index_col=0)
formula = pd.read_csv("cleaned_data.csv")["formula"]

def canon(f):
    d = {}
    for el, n in re.findall(r"([A-Z][a-z]?)(\d+(?:\.\d+)?)?", f):
        d[el] = d.get(el, 0) + (float(n) if n else 1.0)
    total = sum(d.values())
    return "_".join(f"{el}{d[el] / total:.3f}" for el in sorted(d))

group = formula.map(canon)
strat = np.select(
    [labels['Laves'] == 1, labels['B2'] == 1, labels['Sec'] == 1, labels['FCC'] == 1],
    ['Laves', 'B2', 'Sec', 'FCC'],
    default='BCC'
)

def make_fold(seed):
    f = pd.Series(-1, index=feature.index)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    for i, (_, te) in enumerate(splitter.split(feature, strat, groups=group)):
        f.iloc[te] = i
    assert (f == -1).sum() == 0
    assert pd.DataFrame({"g": group, "f": f}).groupby("g")["f"].nunique().max() == 1
    return f

phases = ["BCC", "FCC", "B2", "Laves", "Sec"]
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)
Y = labels[phases]
models = {
    "dummy": DummyClassifier(strategy="prior"),
    "logreg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    "rf_balanced": RandomForestClassifier(random_state=42, class_weight="balanced"),
    "rf": RandomForestClassifier(random_state=42),
    "et": ExtraTreesClassifier(random_state=42),
    "et_balanced": ExtraTreesClassifier(random_state=42, class_weight="balanced"),
}

seeds = [0, 1, 2, 3, 4]
folds = {s: make_fold(s) for s in seeds}

physics = ["delta", "delta_H", "delta_S", "mean_valence_electrons", "mean_melting_point", "delta_chi"]
proc = [c for c in feature.columns if c.startswith("proc_")]
model_cols = physics + ["h_min_pair"] + proc

def evaluate_phase(model, phase, fold, cols=None):
    X = feature if cols is None else feature[cols]
    oof = pd.Series(0.0, index=feature.index)
    for k in range(5):
        train, test = fold != k, fold == k
        m = clone(model).fit(X[train], Y.loc[train, phase])
        oof[test] = m.predict_proba(X[test])[:, 1]
    df = pd.DataFrame({"group": group, "true": Y[phase], "score": oof})
    pa = df.groupby("group").agg(true=("true", "mean"), score=("score", "mean"))
    true = (pa["true"] >= 0.5).astype(int)
    return average_precision_score(true, pa["score"]), true.mean()

feature_sets = {
    "logreg_all": ("logreg", None),
    "rf_all": ("rf_balanced", None),
    "rf_physics": ("rf_balanced", physics),
    "rf_physics_proc": ("rf_balanced", physics + proc),
}
rows = []
for fs, (name, cols) in feature_sets.items():
    for phase in phases:
        for s in seeds:
            ap, prev = evaluate_phase(models[name], phase, folds[s], cols)
            rows.append({"fs": fs, "phase": phase, "seed": s, "ap": ap, "prevalence": prev})
ap = pd.DataFrame(rows)
print(ap.groupby(["phase", "fs"])["ap"].agg(["mean", "std"]).unstack().round(3).loc[phases])
print(ap.groupby("phase")["prevalence"].mean().round(3).loc[phases])

# model vs descriptor: unweighted RF and extra trees, with and without h_min_pair
attribution = {
    "rf_physics_proc": ("rf", physics + proc),
    "rf_physics_hmin_proc": ("rf", model_cols),
    "et_physics_proc": ("et", physics + proc),
    "et_balanced_physics_hmin_proc": ("et_balanced", model_cols),
    "et_physics_hmin_proc": ("et", model_cols),
}
rows = []
for fs, (name, cols) in attribution.items():
    for phase in phases:
        for s in seeds:
            rows.append({"fs": fs, "phase": phase, "seed": s, "ap": evaluate_phase(models[name], phase, folds[s], cols)[0]})
print(pd.DataFrame(rows).groupby(["phase", "fs"])["ap"].agg(["mean", "std"]).unstack().round(3).loc[phases])

def ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    b = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    return sum(abs(y[b == i].mean() - p[b == i].mean()) * (b == i).mean() for i in range(bins) if (b == i).any())

# class weighting for the final extra trees: alloy-level AP and record-level Brier / ECE
rows = []
for name in ["et_balanced", "et"]:
    for phase in phases:
        for s in seeds:
            oof = pd.Series(0.0, index=feature.index)
            for k in range(5):
                train, test = folds[s] != k, folds[s] == k
                m = clone(models[name]).fit(feature.loc[train, model_cols], Y.loc[train, phase])
                oof[test] = m.predict_proba(feature.loc[test, model_cols])[:, 1]
            y = Y[phase].values
            rows.append({"model": name, "phase": phase, "seed": s, "ap": evaluate_phase(models[name], phase, folds[s], model_cols)[0],
                         "brier": np.mean((oof.values - y) ** 2), "ece": ece(y, oof.values), "mean_p": oof.mean()})
print(pd.DataFrame(rows).groupby(["model", "phase"])[["ap", "brier", "ece", "mean_p"]].mean().round(3))

def permutation_importance_phase(model, phase, fold, cols, permute=None, blocks=None, seed=0):
    # drop in alloy-level AP when one descriptor is shuffled across the alloys of each test fold;
    # one value per composition, applied to all its rows, so repeated alloys are not over-weighted
    # (descriptors are identical within a composition up to rounding of the formula).
    # blocks: groups of columns that vary within a composition (the processing flags); each block
    # is shuffled jointly across the records of the test fold, with its own random stream
    rng = np.random.default_rng(seed)
    rng_blocks = np.random.default_rng(seed + 1)
    blocks = blocks or {}
    X = feature[cols]
    base = pd.Series(0.0, index=feature.index)
    permute = cols if permute is None else permute
    perm = {c: pd.Series(0.0, index=feature.index) for c in list(permute) + list(blocks)}
    for k in range(5):
        train, test = fold != k, fold == k
        m = clone(model).fit(X[train], Y.loc[train, phase])
        base[test] = m.predict_proba(X[test])[:, 1]
        for c in permute:
            Xp = X[test].copy()
            per_alloy = Xp[c].groupby(group[test]).first()
            shuffled = pd.Series(rng.permutation(per_alloy.values), index=per_alloy.index)
            Xp[c] = group[test].map(shuffled).values
            perm[c][test] = m.predict_proba(Xp)[:, 1]
        for name, bcols in blocks.items():
            Xp = X[test].copy()
            Xp[bcols] = Xp[bcols].values[rng_blocks.permutation(len(Xp))]
            perm[name][test] = m.predict_proba(Xp)[:, 1]
    true = (Y[phase].groupby(group).mean() >= 0.5).astype(int)
    ap0 = average_precision_score(true, base.groupby(group).mean().loc[true.index])
    return {c: ap0 - average_precision_score(true, perm[c].groupby(group).mean().loc[true.index]) for c in perm}

# final model: unweighted extra trees on the 12 inputs; the seven descriptors one at a time,
# the five processing flags together as one "Process" input
descriptors = physics + ["h_min_pair"]
imp = pd.DataFrame({p: permutation_importance_phase(models["et"], p, folds[0], model_cols, descriptors, {"Process": proc})
                    for p in phases})
print(imp.round(3))

names = {
    "delta": r"$\delta$ (size mismatch)",
    "delta_H": r"$\Delta H_{\mathrm{mix}}$",
    "delta_S": r"$\Delta S_{\mathrm{mix}}$",
    "mean_valence_electrons": "VEC",
    "mean_melting_point": r"$T_m$",
    "delta_chi": r"$\Delta\chi$",
    "h_min_pair": r"$\Delta H_{\mathrm{pair,min}}$",
    "Process": "Process (5 flags)",
}
blues = LinearSegmentedColormap.from_list("blues", ["#ffffff", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
fig, ax = plt.subplots(figsize=(7, 5.1))
ax.imshow(imp.clip(lower=0).values, cmap=blues, vmin=0, vmax=imp.values.max(), aspect="auto")
for i in range(imp.shape[0]):
    for j in range(imp.shape[1]):
        v = imp.values[i, j]
        ax.text(j, i, f"{0.0 if abs(v) < 0.005 else v:.2f}", ha="center", va="center", fontsize=9,
                color="white" if v > 0.6 * imp.values.max() else "#1f2937")
ax.set_xticks(range(len(phases)), ["BCC", "FCC", "B2", "Laves", "Secondary"])
ax.set_yticks(range(len(imp.index)), [names[d] for d in imp.index])
ax.tick_params(length=0)
for sp in ax.spines.values():
    sp.set_visible(False)
ax.set_title("Permutation importance, final extra trees model\n(drop in held-out alloy-level AP, seed 0)", fontsize=10)
fig.tight_layout()
fig.savefig("research/feature_importance.png", dpi=150)

combo = Y.apply(lambda r: "+".join([p for p in phases if r[p] == 1]) or "none", axis=1)
print(combo.value_counts())

def evaluate_combo(model, fold, cols, min_count=15):
    vc = combo.value_counts()
    y = combo.where(combo.map(vc) >= min_count, "rare_combo")
    oof = pd.DataFrame(0.0, index=feature.index, columns=sorted(y.unique()))
    for k in range(5):
        train, test = fold != k, fold == k
        m = clone(model).fit(feature.loc[train, cols], y[train])
        oof.loc[test, m.classes_] = m.predict_proba(feature.loc[test, cols])
    pa = oof.groupby(group).mean()
    true = y.groupby(group).agg(lambda s: s.mode().iloc[0]).loc[pa.index]
    cls = np.array(pa.columns)
    ranked = np.argsort(-pa.values, axis=1)
    top1 = (cls[ranked[:, 0]] == true.values).mean()
    top3 = np.mean([t in cls[r[:3]] for t, r in zip(true.values, ranked)])
    baseline = true.value_counts(normalize=True).max()
    onehot = (np.array(true.tolist())[:, None] == cls.astype(str)[None, :]).astype(float)
    brier = ((pa.values - onehot) ** 2).sum(axis=1).mean()
    calib = ece((cls[ranked[:, 0]] == true.values).astype(float), pa.values.max(axis=1))
    return top1, top3, baseline, brier, calib

combo_runs = {
    "rf_physics": ("rf_balanced", physics),
    "rf_physics_proc": ("rf_balanced", physics + proc),
    "et_physics_proc": ("et", physics + proc),
    "et_balanced_physics_hmin_proc": ("et_balanced", model_cols),
    "et_physics_hmin_proc": ("et", model_cols),
}
rows = []
for fs, (name, cols) in combo_runs.items():
    for s in seeds:
        t1, t3, b, br, ce = evaluate_combo(models[name], folds[s], cols)
        rows.append({"fs": fs, "seed": s, "top1": t1, "top3": t3, "baseline": b, "brier": br, "ece": ce})

print(pd.DataFrame(rows).groupby("fs")[["top1", "top3", "baseline", "brier", "ece"]].agg(["mean", "std"]).round(3))