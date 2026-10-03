# Overall permutation importance of the final model: which input matters most across all phases.
# Two views, both on held-out folds of the grouped CV (split seeds 0-4, as in modelling_multilabel.py):
#   1. the phase-combination model (unweighted extra trees, 12 inputs): drop in alloy-level top-1 accuracy
#   2. the five per-phase models: drop in alloy-level AP, averaged over BCC, FCC, B2, Laves and Sec
# Descriptors are shuffled across alloys (one value per composition); the five processing flags are
# shuffled together as one input (Process) across records, as in research/feature_importance.png.
import re
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.ensemble import ExtraTreesClassifier
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
    return f

phases = ["BCC", "FCC", "B2", "Laves", "Sec"]
Y = labels[phases]
model = ExtraTreesClassifier(random_state=42)
seeds = [0, 1, 2, 3, 4]
physics = ["delta", "delta_H", "delta_S", "mean_valence_electrons", "mean_melting_point", "delta_chi"]
proc = [c for c in feature.columns if c.startswith("proc_")]
descriptors = physics + ["h_min_pair"]
model_cols = descriptors + proc
inputs = descriptors + ["Process"]

combo = Y.apply(lambda r: "+".join([p for p in phases if r[p] == 1]) or "none", axis=1)
vc = combo.value_counts()
combo = combo.where(combo.map(vc) >= 15, "rare_combo")

def shuffled_inputs(Xt, gt, rng, rng_blocks):
    # yields (input name, permuted copy of the test fold)
    for c in descriptors:
        Xp = Xt.copy()
        per_alloy = Xp[c].groupby(gt).first()
        shuffled = pd.Series(rng.permutation(per_alloy.values), index=per_alloy.index)
        Xp[c] = gt.map(shuffled).values
        yield c, Xp
    Xp = Xt.copy()
    Xp[proc] = Xp[proc].values[rng_blocks.permutation(len(Xp))]
    yield "Process", Xp

def oof_scores(y, fold, seed, proba):
    # out-of-fold predictions for the unshuffled test folds and for each shuffled input
    rng, rng_blocks = np.random.default_rng(seed), np.random.default_rng(seed + 1)
    X = feature[model_cols]
    out = {name: None for name in ["base"] + inputs}
    for k in range(5):
        train, test = fold != k, fold == k
        m = clone(model).fit(X[train], y[train])
        preds = [("base", X[test])] + list(shuffled_inputs(X[test], group[test], rng, rng_blocks))
        for name, Xp in preds:
            p = pd.DataFrame(proba(m, Xp), index=Xp.index)
            out[name] = p if out[name] is None else pd.concat([out[name], p])
    return {name: p.loc[feature.index] for name, p in out.items()}

def phase_drop(phase, fold, seed):
    true = (Y[phase].groupby(group).mean() >= 0.5).astype(int)
    s = oof_scores(Y[phase], fold, seed, lambda m, X: m.predict_proba(X)[:, 1])
    ap = {n: average_precision_score(true, p[0].groupby(group).mean().loc[true.index]) for n, p in s.items()}
    return {n: ap["base"] - ap[n] for n in inputs}

def combo_drop(fold, seed):
    cls = sorted(combo.unique())
    def proba(m, X):
        out = np.zeros((len(X), len(cls)))
        out[:, [cls.index(c) for c in m.classes_]] = m.predict_proba(X)
        return out
    s = oof_scores(combo, fold, seed, proba)
    true = combo.groupby(group).agg(lambda v: v.mode().iloc[0])
    acc = {}
    for n, p in s.items():
        pa = p.groupby(group).mean().loc[true.index]
        acc[n] = (np.array(cls)[pa.values.argmax(axis=1)] == true.values).mean()
    return {n: acc["base"] - acc[n] for n in inputs}, acc["base"]

phase_rows, combo_rows = [], []
for s in seeds:
    fold = make_fold(s)
    for phase in phases:
        phase_rows.append({"seed": s, "phase": phase, **phase_drop(phase, fold, s)})
    drop, base = combo_drop(fold, s)
    combo_rows.append({"seed": s, "base_top1": base, **drop})

per_phase = pd.DataFrame(phase_rows)
print("seed 0, per phase (compare with the README table):")
print(per_phase[per_phase.seed == 0].set_index("phase")[inputs].T.round(3))
phase_mean = per_phase.groupby("seed")[inputs].mean()  # average over the five phases, per seed
combo_df = pd.DataFrame(combo_rows).set_index("seed")
print("combination model base top-1:", combo_df["base_top1"].round(3).tolist())
summary = pd.DataFrame({
    "combo_top1_drop": combo_df[inputs].mean(), "combo_sd": combo_df[inputs].std(),
    "phase_mean_ap_drop": phase_mean.mean(), "phase_sd": phase_mean.std(),
}).sort_values("combo_top1_drop", ascending=False)
print(summary.round(3))

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
order = summary.index[::-1]
fig, axes = plt.subplots(1, 2, figsize=(9, 4.4), sharey=True)
panels = [
    ("combo_top1_drop", "combo_sd", "Phase-combination model\ndrop in top-1 accuracy"),
    ("phase_mean_ap_drop", "phase_sd", "Five phase models\nmean drop in AP"),
]
for ax, (col, sd, title) in zip(axes, panels):
    colors = ["#8a94a6" if n == "Process" else "#3987e5" for n in order]
    ax.barh(range(len(order)), summary.loc[order, col], xerr=summary.loc[order, sd], color=colors,
            height=0.65, error_kw={"ecolor": "#1f2937", "elinewidth": 0.8, "capsize": 2})
    for i, v in enumerate(summary.loc[order, col]):
        ax.text(max(v, 0) + summary.loc[order, sd].iloc[i] + 0.004, i, f"{v:.3f}", va="center", fontsize=8, color="#1f2937")
    ax.set_title(title, fontsize=10)
    ax.axvline(0, color="#1f2937", linewidth=0.8)
    ax.set_xlim(min(0, summary[col].min() - 0.01), (summary[col] + summary[sd]).max() * 1.25)
    ax.tick_params(axis="y", length=0)
    for sp in ["top", "right", "left"]:
        ax.spines[sp].set_visible(False)
axes[0].set_yticks(range(len(order)), [names[n] for n in order])
fig.suptitle("Overall permutation importance, final extra trees model (held-out, split seeds 0-4, mean ± sd)", fontsize=10)
fig.tight_layout()
fig.savefig("research/feature_importance_overall.png", dpi=150)
