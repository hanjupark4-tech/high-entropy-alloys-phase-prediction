import re
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, recall_score

feature = pd.read_csv("features.csv", index_col=0)
target = pd.read_csv("target.csv", index_col=0).squeeze()
formula = pd.read_csv("cleaned_data.csv")["formula"]
labels = ["BCC", "FCC", "other"]


def canon(f):
    d = {}
    for el, n in re.findall(r"([A-Z][a-z]?)(\d+(?:\.\d+)?)?", f):
        d[el] = d.get(el, 0) + (float(n) if n else 1.0)
    total = sum(d.values())
    return "_".join(f"{el}{d[el] / total:.3f}" for el in sorted(d))


group = formula.map(canon)


def make_fold(seed):
    f = pd.Series(-1, index=feature.index)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    for i, (_, te) in enumerate(splitter.split(feature, target, groups=group)):
        f.iloc[te] = i
    assert (f == -1).sum() == 0
    assert pd.DataFrame({"g": group, "f": f}).groupby("g")["f"].nunique().max() == 1
    return f


def evaluate(name, model, fold):
    oof = pd.Series(index=feature.index, dtype=object)
    fold_f1 = []
    for k in range(5):
        train, test = fold != k, fold == k
        m = clone(model).fit(feature[train], target[train])
        pred = m.predict(feature[test])
        oof[test] = pred
        fold_f1.append(f1_score(target[test], pred, average="macro"))

    fe = pd.DataFrame({"group": group, "true": target, "pred": oof})
    pa = fe.groupby("group").agg(
        true=("true", lambda s: s.mode()[0]),
        pred=("pred", lambda s: s.mode()[0]),
    )
    rec = recall_score(pa["true"], pa["pred"], labels=labels, average=None)
    row = {
        "model": name,
        "row_acc": accuracy_score(target, oof),
        "row_f1": np.mean(fold_f1),
        "alloy_acc": accuracy_score(pa["true"], pa["pred"]),
        "alloy_f1": f1_score(pa["true"], pa["pred"], average="macro"),
        "rec_BCC": rec[0],
        "rec_FCC": rec[1],
        "rec_other": rec[2],
    }
    cm = confusion_matrix(pa["true"], pa["pred"], labels=labels)
    return row, cm


models = {
    "dummy": DummyClassifier(strategy="most_frequent"),
    "logreg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    "rf_balanced": RandomForestClassifier(random_state=42, class_weight="balanced"),
}

results = []
final_cm = None
for seed in range(5):
    fold = make_fold(seed)
    for name, model in models.items():
        row, cm = evaluate(name, model, fold)
        row["seed"] = seed
        results.append(row)
        if name == "rf_balanced" and seed == 0:
            final_cm = cm

res = pd.DataFrame(results)
cols = ["row_acc", "row_f1", "alloy_acc", "alloy_f1", "rec_BCC", "rec_FCC", "rec_other"]
print(res.groupby("model")[cols].agg(["mean", "std"]).round(3).to_string())
print(pd.DataFrame(final_cm, index=labels, columns=labels))

final_model = clone(models["rf_balanced"]).fit(feature, target)
importance = pd.Series(final_model.feature_importances_, index=feature.columns)
importance = importance.sort_values(ascending=False)
print(importance.head(15))

top = importance.head(15).sort_values()
plt.figure(figsize=(7, 5))
top.plot.barh()
plt.xlabel("Feature importance")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)