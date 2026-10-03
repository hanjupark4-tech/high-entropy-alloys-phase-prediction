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
| BCC | 0.628 | 0.939 ± 0.005 | 0.985 ± 0.001 | 0.975 ± 0.003 | 0.974 ± 0.005 |
| FCC | 0.471 | 0.937 ± 0.002 | 0.981 ± 0.003 | 0.969 ± 0.007 | 0.975 ± 0.009 |
| B2 | 0.148 | 0.583 ± 0.012 | 0.850 ± 0.023 | 0.780 ± 0.017 | 0.830 ± 0.025 |
| Laves | 0.071 | 0.384 ± 0.031 | 0.594 ± 0.035 | 0.474 ± 0.020 | 0.552 ± 0.036 |
| Secondary phase | 0.305 | 0.630 ± 0.011 | 0.854 ± 0.014 | 0.825 ± 0.013 | 0.839 ± 0.015 |

- **Six physics descriptors recover almost all of the BCC and FCC signal.** They come within 0.012 AP of the full 43-feature model.
- **The gap grows for harder phases.** It is 0.03 to 0.07 for B2 and secondary phases and 0.12 for Laves. Laves formation probably depends on which elements are present, not only on bulk descriptors. Adding the processing route recovers part of the gap (0.47 to 0.55).
- **Laves is the hardest phase.** It appears in about 7% of alloys and its AP varies the most across seeds.

Permutation importance of each physics descriptor (random forest, drop in AP when the feature is shuffled, seed 0):

| Descriptor | BCC | FCC | B2 | Laves | Secondary |
|---|---|---|---|---|---|
| delta (size mismatch) | 0.009 | 0.008 | 0.181 | 0.077 | 0.122 |
| delta H mix | 0.007 | 0.007 | 0.258 | 0.196 | 0.063 |
| delta S mix | 0.009 | 0.006 | 0.148 | -0.021 | 0.119 |
| VEC | 0.120 | 0.286 | 0.209 | 0.165 | 0.100 |
| mean melting point | 0.030 | 0.017 | 0.258 | 0.175 | 0.108 |
| delta chi | 0.011 | 0.009 | 0.096 | 0.029 | 0.107 |

VEC alone drives BCC and FCC, matching the classical VEC rule. B2 and Laves depend on mixing enthalpy and melting point, while size mismatch matters less for Laves here. Correlated descriptors share importance, so a low value does not prove a feature is unimportant.

### Phase combinations

Each distinct combination of the five phases is treated as one class (for example `BCC+FCC` or `BCC+Laves`). Combinations with fewer than 15 records are merged into one `rare_combo` class, which leaves 11 classes. The most frequent are BCC (488 records), FCC (273), BCC+Sec (104), FCC+Sec (100) and BCC+FCC (94).

| Inputs | Top-1 accuracy | Top-3 accuracy |
|---|---|---|
| 6 physics descriptors | 0.652 ± 0.009 | 0.904 ± 0.013 |
| 6 physics descriptors + processing | 0.676 ± 0.013 | 0.910 ± 0.013 |
| Majority-class baseline | 0.259 | |

### Baseline: BCC / FCC / other

The first version of this project predicted three classes. Alloy-level results with a random forest (balanced) and 42 features, before adding mixing enthalpy:

| Model | Row macro F1 | Alloy accuracy | Alloy macro F1 | FCC recall | Other recall |
|---|---|---|---|---|---|
| Majority-class baseline | 0.214 ± 0.000 | 0.615 | 0.254 | 0.00 | 1.00 |
| Logistic regression | 0.686 ± 0.021 | 0.724 ± 0.007 | 0.675 ± 0.009 | 0.574 ± 0.021 | 0.777 ± 0.004 |
| **Random forest (balanced)** | **0.722 ± 0.018** | **0.786 ± 0.013** | **0.748 ± 0.014** | 0.652 ± 0.025 | 0.823 ± 0.017 |

Adding mixing enthalpy left the random forest unchanged (alloy macro F1 0.748 to 0.748) and improved logistic regression (0.675 to 0.700, alloy accuracy 0.724 to 0.749). The 3-class setup hides most of the structure, since `other` lumps together B2, Laves and secondary-phase alloys, which is why the project moved to multi-label prediction.

![Feature importance](research/feature_importance.png)

Impurity-based feature importance of the 3-class random forest, trained on the 11 model inputs. The five processing flags are summed into one bar (Process). VEC contributes the most (about 0.28), followed by mean melting point (about 0.19), size mismatch (about 0.15) and mixing enthalpy (about 0.14). Processing as a whole contributes about 0.06. These values come from one model fitted on all data and are less reliable than the permutation importances above, which are computed on held-out folds.

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
| `research/modelling.py` | 3-class baseline with grouped cross-validation, and the feature importance figure for the 11 model inputs |
| `research/modelling_multilabel.py` | Multi-label evaluation: per-phase average precision for all four models, permutation importance of the physics descriptors, and the phase-combination comparison (physics only vs physics + processing) |
| `research/leakage.py` | Random vs composition-grouped cross-validation for the per-phase and combination random forests, writes `leakage_results.csv` |
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

## How much does a random split inflate the scores?

`research/leakage.py` reruns the random forest models with a plain random split and compares them to the grouped split. Both use 5 folds, the same 5 seeds, the same stratification label and the same alloy-level metrics. The only difference is that the random split (`StratifiedKFold`) ignores composition, so on average 72% of test rows have the same composition in the training folds (0% for the grouped split). The gap is random minus grouped, mean ± std over the 5 seeds.

| Phase | Prevalence | RF all 43 features: grouped | random | gap | RF physics + processing: grouped | random | gap |
|---|---|---|---|---|---|---|---|
| BCC | 0.628 | 0.985 ± 0.001 | 0.991 ± 0.001 | +0.006 ± 0.001 | 0.974 ± 0.005 | 0.988 ± 0.003 | +0.014 ± 0.007 |
| FCC | 0.471 | 0.981 ± 0.003 | 0.989 ± 0.005 | +0.008 ± 0.006 | 0.975 ± 0.009 | 0.981 ± 0.009 | +0.006 ± 0.014 |
| B2 | 0.148 | 0.850 ± 0.023 | 0.902 ± 0.012 | +0.053 ± 0.022 | 0.830 ± 0.025 | 0.893 ± 0.022 | +0.063 ± 0.017 |
| Laves | 0.071 | 0.594 ± 0.035 | 0.892 ± 0.020 | **+0.299 ± 0.033** | 0.552 ± 0.036 | 0.859 ± 0.021 | **+0.307 ± 0.044** |
| Secondary phase | 0.305 | 0.854 ± 0.014 | 0.906 ± 0.006 | +0.051 ± 0.012 | 0.839 ± 0.015 | 0.898 ± 0.009 | +0.059 ± 0.016 |

With the 6 physics descriptors alone the gaps are BCC +0.013, FCC +0.010, B2 +0.097, Laves +0.389 and secondary phase +0.070 (Laves AP 0.474 grouped vs 0.863 random).

| Phase combinations | Grouped | Random | Gap |
|---|---|---|---|
| Physics, top-1 | 0.652 ± 0.009 | 0.741 ± 0.007 | +0.089 ± 0.009 |
| Physics, top-3 | 0.904 ± 0.013 | 0.934 ± 0.007 | +0.031 ± 0.013 |
| Physics + processing, top-1 | 0.676 ± 0.013 | 0.764 ± 0.004 | +0.088 ± 0.010 |
| Physics + processing, top-3 | 0.910 ± 0.013 | 0.944 ± 0.012 | +0.034 ± 0.009 |

- **Leakage is small for BCC and FCC and large for the rare phases.** BCC and FCC gain under 0.015 AP, close to seed noise, because VEC separates them well for unseen alloys too. B2 and secondary phases gain 0.05 to 0.10.
- **A random split makes Laves look solved.** Laves AP rises from about 0.5 to about 0.86 to 0.89, so most of the apparent Laves skill under a random split is memorising repeated alloys.
- **Combination accuracy is inflated by about 9 points at top-1** (0.68 to 0.76 with processing).
- **Removing element fractions does not remove the leakage.** With the 6 physics descriptors only, the random split lifts Laves by 0.39 and B2 by 0.10, more than with all 43 features, because six continuous descriptors are still enough to recognise a repeated composition.

## Limitations

- Composition and a coarse processing category are the only inputs. Heat treatment, which controls precipitation of secondary phases, is not captured.
- Laves and B2 predictions are the least reliable, and Laves AP varies the most across seeds.
- 32 formulas carry conflicting labels across records (different processing, or label noise), which caps achievable accuracy.
- Compositions that differ only slightly (for example Al0.3 and Al0.304) are treated as different alloys and can still fall on opposite sides of a split.
- 19 element pairs have no tabulated Miedema value and are set to zero when computing mixing enthalpy.
- Random forest probabilities are not calibrated, and the permutation importances above come from a single seed.
- VEC values are hand-entered and follow one literature convention.

## Possible extensions

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
python research/leakage.py
python train_final.py
streamlit run app.py
```

Run every command from the repository root. The numbers in this README were produced with scikit-learn 1.9.1, the version pinned in `requirements.txt`. Random forest scores shift slightly across scikit-learn versions.
