import re
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone
from sklearn.metrics import average_precision_score

# Random K-fold vs composition-grouped K-fold, same models, seeds and metrics
# as research/modelling_multilabel.py. The gap is how much a random split
# inflates the scores through repeated compositions.

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

def make_fold(seed, grouped):
    f = pd.Series(-1, index=feature.index)
    if grouped:
        splits = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed).split(feature, strat, groups=group)
    else:
        splits = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed).split(feature, strat)
    for i, (_, te) in enumerate(splits):
        f.iloc[te] = i
    assert (f == -1).sum() == 0
    if grouped:
        assert pd.DataFrame({"g": group, "f": f}).groupby("g")["f"].nunique().max() == 1
    return f

def seen_in_train(fold):
    # share of test rows whose composition also appears in the training folds
    seen = [group[fold == k].isin(set(group[fold != k])) for k in range(5)]
    return pd.concat(seen).mean()

phases = ["BCC", "FCC", "B2", "Laves", "Sec"]
Y = labels[phases]
rf = RandomForestClassifier(random_state=42, class_weight="balanced", n_jobs=-1)

seeds = [0, 1, 2, 3, 4]
splits = {"random": False, "grouped": True}
folds = {(sp, s): make_fold(s, g) for sp, g in splits.items() for s in seeds}

physics = ["delta", "delta_H", "delta_S", "mean_valence_electrons", "mean_melting_point", "delta_chi"]
proc = [c for c in feature.columns if c.startswith("proc_")]
feature_sets = {"all": list(feature.columns), "physics": physics, "physics_proc": physics + proc}

def evaluate_phase(model, phase, fold, cols):
    oof = pd.Series(0.0, index=feature.index)
    for k in range(5):
        train, test = fold != k, fold == k
        m = clone(model).fit(feature.loc[train, cols], Y.loc[train, phase])
        oof[test] = m.predict_proba(feature.loc[test, cols])[:, 1]
    df = pd.DataFrame({"group": group, "true": Y[phase], "score": oof})
    pa = df.groupby("group").agg(true=("true", "mean"), score=("score", "mean"))
    true = (pa["true"] >= 0.5).astype(int)
    return average_precision_score(true, pa["score"])

combo = Y.apply(lambda r: "+".join([p for p in phases if r[p] == 1]) or "none", axis=1)

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
    return top1, top3

for sp in splits:
    print(f"{sp}: test rows whose composition is in train = {np.mean([seen_in_train(folds[(sp, s)]) for s in seeds]):.3f}")

rows = []
for fs, cols in feature_sets.items():
    for phase in phases:
        for sp in splits:
            for s in seeds:
                ap = evaluate_phase(rf, phase, folds[(sp, s)], cols)
                rows.append({"task": phase, "fs": fs, "metric": "AP", "split": sp, "seed": s, "score": ap})
for fs in ["physics", "physics_proc"]:
    for sp in splits:
        for s in seeds:
            t1, t3 = evaluate_combo(rf, folds[(sp, s)], feature_sets[fs])
            rows.append({"task": "combo", "fs": fs, "metric": "top1", "split": sp, "seed": s, "score": t1})
            rows.append({"task": "combo", "fs": fs, "metric": "top3", "split": sp, "seed": s, "score": t3})

res = pd.DataFrame(rows)
wide = res.pivot_table(index=["task", "fs", "metric", "seed"], columns="split", values="score")
wide["gap"] = wide["random"] - wide["grouped"]
summary = wide.groupby(["task", "fs", "metric"], sort=False).agg(["mean", "std"]).round(3)
order = [(p, fs, "AP") for fs in feature_sets for p in phases] + \
        [("combo", fs, m) for fs in ["physics", "physics_proc"] for m in ["top1", "top3"]]
summary = summary.loc[order]
pd.set_option("display.width", 200)
print(summary)
summary.to_csv("leakage_results.csv")
