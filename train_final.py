import joblib
import pandas as pd
import sklearn
from sklearn.ensemble import ExtraTreesClassifier

from featurize import MODEL_COLUMNS, SYMBOLS, composition_key, featurize_table

PHASES = ["BCC", "FCC", "B2", "Laves", "Sec"]
MIN_COMBO = 15

data = pd.read_csv("cleaned_data.csv")
X = featurize_table(data["formula"], data["processing method"])
Xm = X[MODEL_COLUMNS]

micro = data["microstructure"].str.get_dummies(sep="+").rename(columns={"Sec.": "Sec"})
Y = micro.reindex(columns=PHASES, fill_value=0)
combo = Y.apply(lambda r: "+".join(p for p in PHASES if r[p] == 1) or "none", axis=1)
counts = combo.value_counts()
combo_label = combo.where(combo.map(counts) >= MIN_COMBO, "rare_combo")

keys = data["formula"].map(composition_key)
first = ~keys.duplicated()

phase_models = {}
for p in PHASES:
    phase_models[p] = ExtraTreesClassifier(random_state=42).fit(Xm, Y[p])
combo_model = ExtraTreesClassifier(random_state=42).fit(Xm, combo_label)

meta = {
    "n_rows": int(len(data)),
    "n_alloys": int(keys.nunique()),
    "n_elements_range": [int(X["n_elements"].min()), int(X["n_elements"].max())],
    "phases": PHASES,
    "phase_counts": {p: int(Y[p].sum()) for p in PHASES},
    "combo_classes": [str(c) for c in combo_model.classes_],
    "combo_counts": {str(c): int((combo_label == c).sum()) for c in combo_model.classes_},
    "element_alloys": {s: int((X.loc[first, s] > 0).sum()) for s in SYMBOLS},
    "known_keys": sorted(set(keys)),
    "cv": {
        "folds": "5-fold grouped by composition, 5 seeds",
        "ap": {"BCC": 0.979, "FCC": 0.979, "B2": 0.859, "Laves": 0.657, "Sec": 0.871},
        "prevalence": {"BCC": 0.628, "FCC": 0.471, "B2": 0.148, "Laves": 0.071, "Sec": 0.305},
        "top1": 0.712,
        "top3": 0.913,
        "baseline": 0.259,
    },
    "sklearn_version": sklearn.__version__,
}

joblib.dump({"phase_models": phase_models, "combo_model": combo_model, "meta": meta}, "model.joblib", compress=3)

with open("requirements.txt", "w") as f:
    f.write("streamlit>=1.40\npandas\nnumpy\njoblib\nscikit-learn==" + sklearn.__version__ + "\n")

print("phases:", PHASES)
print("combo classes:", list(combo_model.classes_))
print("alloys:", meta["n_alloys"], "rows:", meta["n_rows"])
print("saved model.joblib with scikit-learn", sklearn.__version__)
