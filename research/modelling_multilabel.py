import re
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
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
}

seeds = [0, 1, 2, 3, 4]
folds = {s: make_fold(s) for s in seeds}

physics = ["delta", "delta_H", "delta_S", "mean_valence_electrons", "mean_melting_point", "delta_chi"]
proc = [c for c in feature.columns if c.startswith("proc_")]

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

def permutation_importance_phase(model, phase, fold, cols, seed=0):
    # drop in alloy-level AP when one descriptor is shuffled within each test fold
    rng = np.random.default_rng(seed)
    X = feature[cols]
    base = pd.Series(0.0, index=feature.index)
    perm = {c: pd.Series(0.0, index=feature.index) for c in cols}
    for k in range(5):
        train, test = fold != k, fold == k
        m = clone(model).fit(X[train], Y.loc[train, phase])
        base[test] = m.predict_proba(X[test])[:, 1]
        for c in cols:
            Xp = X[test].copy()
            Xp[c] = rng.permutation(Xp[c].values)
            perm[c][test] = m.predict_proba(Xp)[:, 1]
    true = (Y[phase].groupby(group).mean() >= 0.5).astype(int)
    ap0 = average_precision_score(true, base.groupby(group).mean().loc[true.index])
    return {c: ap0 - average_precision_score(true, perm[c].groupby(group).mean().loc[true.index]) for c in cols}

imp = pd.DataFrame({p: permutation_importance_phase(models["rf_balanced"], p, folds[0], physics) for p in phases})
print(imp.round(3))

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
    return top1, top3, baseline

rows = []
for name, cols in {"physics": physics, "physics_proc": physics + proc}.items():
    for s in seeds:
        t1, t3, b = evaluate_combo(models["rf_balanced"], folds[s], cols)
        rows.append({"fs": name, "seed": s, "top1": t1, "top3": t3, "baseline": b})

print(pd.DataFrame(rows).groupby("fs")[["top1", "top3", "baseline"]].agg(["mean", "std"]).round(3))