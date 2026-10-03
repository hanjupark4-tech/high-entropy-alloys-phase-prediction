import re
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.base import clone
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss

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
strat = pd.Series(np.select(
    [labels['Laves'] == 1, labels['B2'] == 1, labels['Sec'] == 1, labels['FCC'] == 1],
    ['Laves', 'B2', 'Sec', 'FCC'],
    default='BCC'
), index=feature.index)

phases = ["BCC", "FCC", "B2", "Laves", "Sec"]
Y = labels[phases]
physics = ["delta", "delta_H", "delta_S", "mean_valence_electrons", "mean_melting_point", "delta_chi"]
proc = [c for c in feature.columns if c.startswith("proc_")]
X = feature[physics + proc]

models = {
    "balanced": RandomForestClassifier(random_state=42, class_weight="balanced", n_jobs=-1),
    "unweighted": RandomForestClassifier(random_state=42, n_jobs=-1),
}
seeds = [0, 1, 2, 3, 4]

def grouped_folds(idx, seed):
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    f = pd.Series(-1, index=idx)
    for i, (_, te) in enumerate(splitter.split(X.loc[idx], strat[idx], groups=group[idx])):
        f.iloc[te] = i
    assert (f == -1).sum() == 0
    assert pd.DataFrame({"g": group[idx], "f": f}).groupby("g")["f"].nunique().max() == 1
    return f

def fit_calibrators(score, y):
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(score, y)
    platt = LogisticRegression(C=1e6).fit(score.reshape(-1, 1), y)
    return {"isotonic": iso.predict, "sigmoid": lambda s: platt.predict_proba(s.reshape(-1, 1))[:, 1]}

def oof_scores(model, phase, seed):
    """Outer grouped CV. Calibrators are fitted on grouped inner-CV scores of the training fold only."""
    fold = grouped_folds(X.index, seed)
    out = pd.DataFrame(0.0, index=X.index, columns=["raw", "sigmoid", "isotonic"])
    y = Y[phase]
    for k in range(5):
        train, test = X.index[fold != k], X.index[fold == k]
        inner = grouped_folds(train, seed + 100)
        inner_score = pd.Series(0.0, index=train)
        for j in range(5):
            itr, ite = train[inner != j], train[inner == j]
            m = clone(model).fit(X.loc[itr], y[itr])
            inner_score[ite] = m.predict_proba(X.loc[ite])[:, 1]
        cal = fit_calibrators(inner_score.values, y[train].values)
        m = clone(model).fit(X.loc[train], y[train])
        raw = m.predict_proba(X.loc[test])[:, 1]
        out.loc[test, "raw"] = raw
        for name, f in cal.items():
            out.loc[test, name] = f(raw)
    return out

def ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    b = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    return sum(abs(y[b == i].mean() - p[b == i].mean()) * (b == i).mean() for i in range(bins) if (b == i).any())

def metrics(y, p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    prev = y.mean()
    return {
        "brier": brier_score_loss(y, p),
        "bss": 1 - brier_score_loss(y, p) / (prev * (1 - prev)),
        "logloss": log_loss(y, p, labels=[0, 1]),
        "ece": ece(y, p),
        "ap": average_precision_score(y, p),
        "mean_p": p.mean(),
        "prev": prev,
    }

rows, pooled = [], []
for mname, model in models.items():
    for phase in phases:
        for s in seeds:
            out = oof_scores(model, phase, s)
            y = Y[phase].values
            alloy = out.groupby(group).mean()
            ya = (Y[phase].groupby(group).mean().loc[alloy.index] >= 0.5).astype(int).values
            for method in out.columns:
                rows.append({"model": mname, "phase": phase, "seed": s, "method": method, "level": "row",
                             **metrics(y, out[method].values)})
                rows.append({"model": mname, "phase": phase, "seed": s, "method": method, "level": "alloy",
                             **metrics(ya, alloy[method].values)})
            pooled.append(out.assign(model=mname, phase=phase, seed=s, y=y))
        print("done", mname, phase, flush=True)

res = pd.DataFrame(rows)
res.to_csv("calibration_results.csv", index=False)
pd.concat(pooled).to_csv("calibration_oof.csv")

pd.set_option("display.width", 200)
for level in ["row", "alloy"]:
    summary = res[res.level == level].groupby(["model", "phase", "method"])[["brier", "bss", "logloss", "ece", "ap", "mean_p", "prev"]]
    print(f"\n=== {level} level (mean over seeds) ===")
    print(summary.mean().round(3))
    print(summary.std()[["brier", "ece", "ap"]].round(3))

# Reliability curves, row level, pooled over seeds: unweighted RF vs balanced RF, raw and with sigmoid calibration
P = pd.concat(pooled)
curves = {
    "unweighted RF": ("unweighted", "raw", "#2a78d6"),
    "balanced RF": ("balanced", "raw", "#eb6834"),
    "balanced RF + sigmoid": ("balanced", "sigmoid", "#1baf7a"),
}
fig, axes = plt.subplots(1, 5, figsize=(17, 3.8), sharey=True)
edges = np.linspace(0, 1, 11)
for ax, phase in zip(axes, phases):
    ax.plot([0, 1], [0, 1], color="#a3a29c", lw=1, ls="--")
    for label, (mname, method, c) in curves.items():
        d = P[(P.phase == phase) & (P.model == mname)]
        b = np.clip(np.digitize(d[method], edges[1:-1]), 0, 9)
        g = pd.DataFrame({"b": b, "p": d[method], "y": d["y"]}).groupby("b").agg(p=("p", "mean"), y=("y", "mean"), n=("y", "size"))
        g = g[g.n >= 10]
        ax.plot(g.p, g.y, marker="o", ms=4, lw=2, color=c, label=label)
    ax.set_title(f"{phase} (prevalence {d.y.mean():.2f})", fontsize=10)
    ax.set_xlabel("Predicted probability")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.grid(color="#e8e7e2", lw=0.6)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
axes[0].set_ylabel("Observed frequency")
axes[0].legend(frameon=False, fontsize=8, loc="upper left")
fig.suptitle("Reliability of random forest phase probabilities, grouped 5-fold CV, 5 seeds pooled", fontsize=11)
fig.tight_layout()
fig.savefig("research/calibration_reliability.png", dpi=150)
