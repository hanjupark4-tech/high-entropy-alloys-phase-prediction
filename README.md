# High-Entropy Alloy Phase Prediction

Predicting which phases a high-entropy alloy forms (BCC, FCC, B2, Laves, secondary phase) and which phase combinations are most likely, from its composition and processing route alone.

Live app: https://high-entropy-alloys-phase-prediction-ntwaaikt4rzo4mdrdari4f.streamlit.app/

The model uses six physics descriptors from Hume-Rothery-style stability rules (atomic size mismatch, mixing enthalpy, mixing entropy, valence electron concentration, mean melting point, electronegativity spread) plus the processing route. It is evaluated with composition-grouped cross-validation, which avoids the data leakage typical of alloy datasets.

## Results

Evaluation: 5-fold `StratifiedGroupKFold`, repeated over 5 random seeds (mean ± std). Folds are grouped by normalised composition, so the same alloy never appears in both train and test. Scores are computed at alloy level: predicted probabilities are averaged over repeated measurements of the same composition (522 unique compositions), and the truth is the majority label.

### Per-phase prediction

One random forest per phase. The metric is average precision (AP). A model with no information scores the phase prevalence, so each AP should be read against that column.

| Phase | Prevalence | Logistic regression, all features | RF, all 43 features | RF, 6 physics descriptors | RF, physics + processing |
|---|---|---|---|---|---|
| BCC | 0.628 | 0.944 ± 0.002 | 0.985 ± 0.001 | 0.972 ± 0.003 | 0.975 ± 0.003 |
| FCC | 0.471 | 0.936 ± 0.011 | 0.984 ± 0.003 | 0.975 ± 0.004 | 0.980 ± 0.002 |
| B2 | 0.148 | 0.564 ± 0.016 | 0.831 ± 0.016 | 0.782 ± 0.029 | 0.806 ± 0.024 |
| Laves | 0.071 | 0.343 ± 0.038 | 0.570 ± 0.062 | 0.426 ± 0.032 | 0.514 ± 0.040 |
| Secondary phase | 0.305 | 0.622 ± 0.016 | 0.845 ± 0.008 | 0.817 ± 0.009 | 0.834 ± 0.010 |

- **Six physics descriptors recover almost all of the BCC and FCC signal.** They come within 0.013 AP of the full 43-feature model.
- **The gap grows for harder phases.** It is 0.03 to 0.05 for B2 and secondary phases and 0.14 for Laves. Laves formation probably depends on which elements are present, not only on bulk descriptors. Adding the processing route recovers part of the gap (0.43 to 0.51).
- **Laves is the hardest phase.** It appears in about 7% of alloys and its AP varies the most across seeds.

Permutation importance of each physics descriptor (random forest, drop in AP when the feature is shuffled, seed 0):

| Descriptor | BCC | FCC | B2 | Laves | Secondary |
|---|---|---|---|---|---|
| delta (size mismatch) | 0.003 | 0.010 | 0.179 | 0.039 | 0.124 |
| delta H mix | 0.001 | 0.011 | 0.235 | 0.209 | 0.083 |
| delta S mix | 0.001 | 0.007 | 0.126 | -0.025 | 0.164 |
| VEC | 0.131 | 0.307 | 0.153 | 0.186 | 0.092 |
| mean melting point | 0.023 | 0.019 | 0.201 | 0.228 | 0.111 |
| delta chi | 0.003 | 0.020 | 0.121 | 0.097 | 0.091 |

VEC alone drives BCC and FCC, matching the classical VEC rule. B2 and Laves depend on mixing enthalpy and melting point, while size mismatch matters little for Laves here. Correlated descriptors share importance, so a low value does not prove a feature is unimportant.

### Phase combinations

Each distinct combination of the five phases is treated as one class (for example `BCC+FCC` or `BCC+Laves`). Combinations with fewer than 15 records are merged into one `rare_combo` class, which leaves 11 classes. The most frequent are BCC (488 records), FCC (273), BCC+Sec (104), FCC+Sec (100) and BCC+FCC (94).

| Inputs | Top-1 accuracy | Top-3 accuracy |
|---|---|---|
| 6 physics descriptors | 0.672 ± 0.007 | 0.910 ± 0.011 |
| 6 physics descriptors + processing | 0.695 ± 0.011 | 0.922 ± 0.009 |
| Majority-class baseline | 0.259 | |

### Baseline: BCC / FCC / other

The first version of this project predicted three classes. Alloy-level results with a random forest (balanced) and 42 features, before adding mixing enthalpy:

| Model | Row macro F1 | Alloy accuracy | Alloy macro F1 | FCC recall | Other recall |
|---|---|---|---|---|---|
| Majority-class baseline | 0.216 ± 0.002 | 0.615 | 0.254 | 0.00 | 1.00 |
| Logistic regression | 0.667 ± 0.009 | 0.719 ± 0.009 | 0.670 ± 0.012 | 0.566 ± 0.042 | 0.763 ± 0.008 |
| **Random forest (balanced)** | **0.724 ± 0.019** | **0.804 ± 0.006** | **0.754 ± 0.012** | 0.564 ± 0.050 | 0.875 ± 0.006 |

Adding mixing enthalpy left the random forest unchanged (alloy macro F1 0.754 to 0.750) and improved logistic regression (0.670 to 0.704, alloy accuracy 0.719 to 0.752). The 3-class setup hides most of the structure, since `other` lumps together B2, Laves and secondary-phase alloys, which is why the project moved to multi-label prediction.

![Feature importance](research/feature_importance.png)

## Web app

`app.py` is a Streamlit app. Enter a composition and a processing route to get a probability for each of the five phases, the three most likely phase combinations, and the six computed descriptors. It also warns when the composition is in the training data (probabilities are then optimistic), when elements are rare in the data, and that Laves and B2 predictions are the least reliable.

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Pipeline

```
High Entropy Alloy Properties.csv
        |  research/data.py
        v
cleaned_data.csv
        |  research/feature.py
        v
features.csv + target.csv
        |  research/label.py
        v
labels.csv
        |  research/modelling_multilabel.py
        v
metrics

cleaned_data.csv --> train_final.py --> model.joblib --> app.py
```

| File | Purpose |
|---|---|
| `research/data.py` | Cleans column names, drops unused columns, removes rows with missing microstructure or processing method |
| `research/feature.py` | Parses formulas into element fractions and computes composition-based descriptors, including mixing enthalpy |
| `research/label.py` | Splits the microstructure string into binary phase labels |
| `research/modelling.py` | 3-class baseline with grouped cross-validation and feature importance |
| `research/modelling_multilabel.py` | Multi-label experiments: per-phase average precision, feature-set comparison, phase combinations |
| `featurize.py` | Computes the same descriptors for a single composition typed into the app |
| `train_final.py` | Trains the five phase models and the combination model on all data and writes `model.joblib` |
| `app.py` | Streamlit app |
| `miedema_matrix.csv` | Symmetric 28 x 28 matrix of pair mixing enthalpies built from `MiedemaLiquidDeltaHf.tsv` |

## Data

- 1354 records, 522 unique compositions after normalising to atomic fractions
- Source: High Entropy Alloy Properties dataset (https://www.kaggle.com/datasets/sethpointaverage/high-entropy-alloys-properties/data)
- Pair mixing enthalpies for the Miedema model come from the `MiedemaLiquidDeltaHf.tsv` data file shipped with matminer
- CSV files are excluded from the repository by `.gitignore`, except `miedema_matrix.csv`. Place the raw CSV in the project folder and run the scripts in order.

Phase labels come from splitting the `microstructure` string on `+`: BCC, FCC, B2, Laves and secondary phase (`Sec.`). HCP, L12 and Other are rare and are not predicted. 31 records carry none of the five labels.

## Features (43)

| Group | Count | Description |
|---|---|---|
| Element fractions | 28 | Molar fraction of each of the 28 elements in the dataset |
| Mean elemental properties | 4 | Composition-weighted mean atomic radius, electronegativity, melting point and VEC |
| Mismatch descriptors | 2 | Atomic size mismatch (delta, %) and electronegativity spread (delta chi) |
| Mixing entropy | 1 | Ideal mixing entropy, -R sum(c ln c) |
| Mixing enthalpy | 1 | Miedema model, delta H mix = sum over pairs of 4 H_ij c_i c_j |
| Number of elements | 1 | Count of elements with non-zero fraction |
| Processing route | 5 | One-hot: cast, wrought, anneal, powder, other |
| Density | 1 | Calculated density from the source dataset |

The final model and the app use only 11 inputs: delta, delta H mix, delta S mix, VEC, mean melting point, delta chi, and the five processing flags. Microstructure, yield strength and hardness are not used as inputs, since microstructure is a finer description of the target and would leak the answer.

## Validation design

The dataset contains many repeated compositions: 831 of 1354 rows repeat a formula already seen, and one alloy (HfNbTaTiZr) alone accounts for 86 rows. A random split would put copies of the same alloy in both train and test and inflate every score.

- Folds are built with `StratifiedGroupKFold`, grouping by composition normalised to atomic fractions (so `Co1 Fe1 Ni1` and `Co2 Fe2 Ni2` count as one alloy). For stratification each alloy gets a single priority label (Laves, B2, Sec, FCC, otherwise BCC).
- Splits are checked with assertions: no composition appears in two folds and every row is assigned.
- Every comparison is repeated over 5 split seeds. Differences smaller than the seed-to-seed standard deviation are treated as noise.
- Feature scaling for logistic regression is fitted inside each training fold through a pipeline.

## Limitations

- Composition and a coarse processing category are the only inputs. Heat treatment, which controls precipitation of secondary phases, is not captured.
- Laves and B2 predictions are the least reliable, and Laves AP varies the most across seeds.
- 32 formulas carry conflicting labels across records (different processing, or label noise), which caps achievable accuracy.
- Compositions that differ only slightly (for example Al0.3 and Al0.304) are treated as different alloys and can still fall on opposite sides of a split.
- 19 element pairs have no tabulated Miedema value and are set to zero when computing mixing enthalpy.
- Random forest probabilities are not calibrated, and the permutation importances above come from a single seed.
- VEC values are hand-entered and follow one literature convention.

## Possible extensions

- Compare random and grouped splits directly to quantify how much leakage inflates published results
- Test whether dropping size mismatch alone changes Laves performance
- Add the Omega parameter and calibrate the probabilities
- Use the same features for hardness and yield strength regression

## Reproduce

```bash
pip install pandas numpy scikit-learn pymatgen matplotlib
python research/data.py
python research/feature.py
python research/label.py
python research/modelling_multilabel.py
python train_final.py
streamlit run app.py
```

Run every command from the repository root.
