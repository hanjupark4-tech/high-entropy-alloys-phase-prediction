import joblib
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier

from featurize import SYMBOLS, composition_key, featurize_table

data = pd.read_csv("cleaned_data.csv")
X = featurize_table(data["formula"], data["processing method"])
y = data["bcc/fcc/other"]
keys = data["formula"].map(composition_key)
first = ~keys.duplicated()

model = RandomForestClassifier(random_state=42, class_weight="balanced")
model.fit(X, y)

meta = {
    "n_rows": int(len(data)),
    "n_alloys": int(keys.nunique()),
    "n_elements_range": [int(X["n_elements"].min()), int(X["n_elements"].max())],
    "class_counts": {c: int((y == c).sum()) for c in model.classes_},
    "class_mean_vec": {c: float(X.loc[y == c, "mean_valence_electrons"].mean()) for c in model.classes_},
    "element_alloys": {s: int((X.loc[first, s] > 0).sum()) for s in SYMBOLS},
    "known_keys": sorted(set(keys)),
    "sklearn_version": sklearn.__version__,
}

joblib.dump({"model": model, "meta": meta}, "model.joblib", compress=3)

with open("requirements.txt", "w") as f:
    f.write("streamlit>=1.40\npandas\nnumpy\njoblib\nscikit-learn==" + sklearn.__version__ + "\n")

print("classes:", list(model.classes_))
print("alloys:", meta["n_alloys"], "rows:", meta["n_rows"])
print("saved model.joblib with scikit-learn", sklearn.__version__)
