# High-Entropy Alloy Phase Prediction

Predicting which phases a high-entropy alloy forms (BCC, FCC, B2, Laves, secondary phase) and which phase combinations are most likely, from its composition and processing route alone.

Live app: https://high-entropy-alloys-phase-prediction-ntwaaikt4rzo4mdrdari4f.streamlit.app/

The model is a set of extra trees classifiers on 12 inputs: seven physics descriptors (atomic size mismatch, mixing enthalpy, mixing entropy, valence electron concentration, mean melting point and electronegativity spread from Hume-Rothery-style stability rules, plus the strongest pair mixing enthalpy among the alloy's elements) and five processing-route flags. It is evaluated with composition-grouped cross-validation, which avoids the data leakage typical of alloy datasets.

## Results

Evaluation: 5-fold `StratifiedGroupKFold`, repeated over 5 random seeds (mean ± std). Folds are grouped by normalised composition, so the same alloy never appears in both train and test. Scores are computed at alloy level: predicted probabilities are averaged over repeated measurements of the same composition (522 unique compositions), and the truth is the majority label.

### Per-phase prediction

One classifier per phase. The metric is average precision (AP). A model with no information scores the phase prevalence, so each AP should be read against that column. The baseline models below are balanced random forests on the six original physics descriptors (no `h_min_pair`); the final model is compared in the next section.

| Phase | Prevalence | Logistic regression, all features | RF, all 44 features | RF, 6 physics descriptors | RF, physics + processing |
|---|---|---|---|---|---|
| BCC | 0.628 | 0.937 ± 0.005 | 0.985 ± 0.002 | 0.975 ± 0.003 | 0.974 ± 0.005 |
| FCC | 0.471 | 0.937 ± 0.002 | 0.980 ± 0.003 | 0.969 ± 0.007 | 0.975 ± 0.009 |
| B2 | 0.148 | 0.583 ± 0.013 | 0.852 ± 0.021 | 0.780 ± 0.017 | 0.830 ± 0.025 |
| Laves | 0.071 | 0.383 ± 0.031 | 0.594 ± 0.038 | 0.474 ± 0.020 | 0.552 ± 0.036 |
| Secondary phase | 0.305 | 0.642 ± 0.010 | 0.859 ± 0.011 | 0.825 ± 0.013 | 0.839 ± 0.015 |

- **The six original physics descriptors recover almost all of the BCC and FCC signal.** They come within 0.011 AP of the full 44-feature model.
- **The gap grows for harder phases.** It is 0.03 to 0.07 for B2 and secondary phases and 0.12 for Laves. Laves formation probably depends on which elements are present, not only on bulk descriptors. Adding the processing route recovers part of the gap (0.47 to 0.55).
- **Laves is the hardest phase.** It appears in about 7% of alloys and its AP varies the most across seeds.

### Final model: extra trees with the strongest pair enthalpy

The final model replaces the random forest with extra trees, which is the main source of the gain on the rare phases. It also adds a seventh descriptor, `h_min_pair`: the most negative Miedema pair enthalpy among the elements present (for example Ni-Zr or Al-Zr). This is a smaller, physically motivated addition. The concentration-weighted delta H mix averages one strongly bonding pair away, while intermetallic formation can hinge on it. The table separates the two effects. All four models are unweighted and use the same folds and seeds as above.

| Phase | RF, 11 inputs | RF + h_min_pair, 12 inputs | ET, 11 inputs | **ET + h_min_pair, 12 inputs (final)** |
|---|---|---|---|---|
| BCC | 0.976 ± 0.004 | 0.978 ± 0.004 | 0.977 ± 0.003 | **0.979 ± 0.003** |
| FCC | 0.975 ± 0.007 | 0.976 ± 0.008 | 0.982 ± 0.005 | **0.979 ± 0.008** |
| B2 | 0.819 ± 0.029 | 0.824 ± 0.027 | 0.856 ± 0.019 | **0.859 ± 0.020** |
| Laves | 0.568 ± 0.027 | 0.630 ± 0.025 | 0.619 ± 0.031 | **0.657 ± 0.029** |
| Secondary phase | 0.839 ± 0.011 | 0.848 ± 0.009 | 0.857 ± 0.016 | **0.871 ± 0.011** |

- **Extra trees drive the Laves and B2 gains.** On 10 further split seeds (5 to 14, model seed varied), extra trees beat the random forest on Laves by +0.065 (9 of 10 seeds) and on B2 by +0.036 (10 of 10). Extra trees pick split thresholds at random, which gives smoother probability rankings for rare classes. Bootstrapping is not the reason: a random forest without bootstrapping scores the same, and extra trees with bootstrapping keep the gain.
- **`h_min_pair` helps the random forest and secondary phases.** It lifts random-forest Laves AP by +0.040 (10 of 10 seeds 5 to 14) and extra-trees secondary-phase AP by +0.020 (10 of 10). On top of extra trees its Laves gain is +0.011 (6 of 10), which is within seed noise, and it adds nothing for B2.
- **Class weighting does not matter for extra trees.** Balanced and unweighted extra trees have the same AP (Laves 0.658 vs 0.657) and nearly the same calibration. Unweighted is slightly better for secondary phases (row-level Brier 0.084 vs 0.091), so the final phase models are unweighted.
- **What did not help Laves:** the Omega parameter, gamma and lambda parameters, VEC spread, radius-ratio descriptors, and gradient-boosted trees (scikit-learn, LightGBM, XGBoost), which scored 0.53 to 0.58. Adding the 28 element fractions raises Laves AP to about 0.72 but makes the model less physics-based, so the app does not use them.

Permutation importance of each physics descriptor (random forest, seed 0): the drop in alloy-level AP when the descriptor is shuffled across the alloys of each held-out fold, with one value per composition so repeated alloys are not over-weighted.

| Descriptor | BCC | FCC | B2 | Laves | Secondary |
|---|---|---|---|---|---|
| delta (size mismatch) | 0.008 | 0.009 | 0.171 | 0.119 | 0.118 |
| delta H mix | 0.003 | 0.004 | 0.189 | 0.182 | 0.100 |
| delta S mix | 0.009 | 0.016 | 0.138 | -0.009 | 0.127 |
| VEC | 0.139 | 0.293 | 0.164 | 0.202 | 0.104 |
| mean melting point | 0.032 | 0.032 | 0.148 | 0.275 | 0.155 |
| delta chi | 0.013 | 0.017 | 0.078 | 0.036 | 0.113 |

VEC alone drives BCC and FCC, matching the classical VEC rule. Laves depends most on melting point, VEC and mixing enthalpy. B2 depends most on mixing enthalpy and size mismatch, but every descriptor contributes. Correlated descriptors share importance, so a low value does not prove a feature is unimportant.

### Probability calibration

The per-phase forests output probabilities, so it matters whether those probabilities match observed frequencies. `research/calibration.py` checks this under the same grouped 5-fold CV and 5 seeds, using the physics + processing inputs. It compares a forest trained with `class_weight="balanced"` against an unweighted forest, each raw and with sigmoid or isotonic calibration. The calibrators are fitted on grouped inner-CV predictions inside each training fold, so no alloy is used both to fit a calibrator and to test it. Brier score and expected calibration error (ECE, 10 bins) are computed per record.

| Phase | Prevalence | RF balanced: mean p, Brier, ECE | RF unweighted: mean p, Brier, ECE | Balanced + sigmoid: Brier, ECE | Balanced + isotonic: Brier, ECE |
|---|---|---|---|---|---|
| BCC | 0.656 | 0.632, 0.061, 0.052 | 0.651, 0.062, 0.053 | 0.060, 0.037 | 0.059, 0.028 |
| FCC | 0.407 | 0.418, 0.042, 0.034 | 0.408, 0.042, 0.039 | 0.043, 0.021 | 0.043, 0.016 |
| B2 | 0.114 | 0.147, 0.058, 0.038 | 0.116, 0.055, 0.030 | 0.055, 0.015 | 0.057, 0.033 |
| Laves | 0.077 | 0.104, 0.052, 0.036 | 0.081, 0.053, 0.022 | 0.054, 0.027 | 0.057, 0.029 |
| Sec | 0.199 | 0.297, 0.106, 0.102 | 0.240, 0.094, 0.066 | 0.091, 0.039 | 0.091, 0.029 |

- **`class_weight="balanced"` inflates minority-phase probabilities under grouped CV.** Secondary-phase probabilities average 0.30 against an observed rate of 0.20 (ECE 0.10), and B2 averages 0.15 against 0.11. BCC and FCC are already close to calibrated.
- **Dropping the class weighting removes most of this bias without hurting ranking.** The unweighted forest's alloy-level AP is 0.976 (BCC), 0.975 (FCC), 0.819 (B2), 0.568 (Laves) and 0.839 (Sec). The balanced forest's AP, shown in the table above, is 0.974, 0.975, 0.830, 0.552 and 0.839.
- **Sigmoid and isotonic calibration hurt Laves.** There are too few Laves alloys to fit a calibrator, so Laves Brier rises from 0.052 to 0.054 (sigmoid) and 0.057 (isotonic), and Laves AP falls from 0.51 to 0.47 and 0.42 at record level. For B2 and secondary phases, calibration lowers ECE further. Both calibrators also cost about 0.02 AP on FCC.

![Reliability curves](research/calibration_reliability.png)

Reliability curves pooled over the 5 seeds, per record. Bins with fewer than 10 records are not shown.

### Check on three unseen alloys

Three annealed NbCrTiAlZr alloys that are not in the training data, all reported as BCC + Laves (Nb is the balance). Each model is trained on all data.

| Alloy (at.%) | Model | BCC | FCC | B2 | Laves | Sec | Top 3 combinations |
|---|---|---|---|---|---|---|---|
| Nb36Cr16Ti16Al16Zr16 | Previous (balanced RF, 11 inputs) | 0.90 | 0.00 | 0.29 | 0.44 | 0.27 | **BCC+Laves 0.35**, BCC 0.28, BCC+Sec 0.16 |
| | ET, 11 inputs | 0.67 | 0.00 | 0.23 | 0.38 | 0.36 | BCC 0.45, other 0.21, BCC+Laves 0.16 |
| | ET + h_min_pair (final) | 0.47 | 0.01 | 0.54 | 0.34 | 0.51 | other 0.55, BCC 0.25, BCC+Laves 0.13 |
| Nb47Cr16Ti16Al5Zr16 | Previous | 0.98 | 0.01 | 0.00 | 0.50 | 0.19 | **BCC+Laves 0.41**, BCC+Sec 0.31, BCC 0.27 |
| | ET, 11 inputs | 0.88 | 0.02 | 0.01 | 0.25 | 0.19 | BCC 0.44, BCC+Laves 0.38, BCC+Sec 0.12 |
| | ET + h_min_pair (final) | 0.77 | 0.03 | 0.22 | 0.42 | 0.25 | BCC 0.35, BCC+Laves 0.29, other 0.14 |
| Nb50Cr16Ti16Al2Zr16 | Previous | 0.99 | 0.02 | 0.02 | 0.39 | 0.33 | BCC+Sec 0.40, BCC 0.29, BCC+Laves 0.25 |
| | ET, 11 inputs | 0.90 | 0.02 | 0.02 | 0.17 | 0.13 | BCC 0.51, BCC+Laves 0.25, BCC+Sec 0.18 |
| | ET + h_min_pair (final) | 0.80 | 0.03 | 0.16 | 0.38 | 0.24 | BCC 0.34, BCC+Laves 0.32, BCC+Sec 0.12 |

The check is neutral to slightly negative for the new model. BCC+Laves is in the top 3 for all three alloys with every model. It is the top combination for two of the three with the previous model and for none with either extra trees model, where it comes second or third behind BCC or `other`. Laves probabilities are not higher than the previous model's (0.34 to 0.42 vs 0.39 to 0.50), although `h_min_pair` keeps them above those of extra trees without it (0.17 to 0.38). For the 16% Al alloy, the final model leans towards B2. Three alloys are an illustration, not a validation.

### Phase combinations

Each distinct combination of the five phases is treated as one class (for example `BCC+FCC` or `BCC+Laves`). Combinations with fewer than 15 records are merged into one `rare_combo` class, which leaves 11 classes. The most frequent are BCC (488 records), FCC (273), BCC+Sec (104), FCC+Sec (100) and BCC+FCC (94).

| Inputs (balanced random forest unless stated) | Top-1 accuracy | Top-3 accuracy |
|---|---|---|
| 6 physics descriptors | 0.652 ± 0.009 | 0.904 ± 0.013 |
| 6 physics descriptors + processing | 0.676 ± 0.013 | 0.910 ± 0.013 |
| Unweighted extra trees, 11 inputs | 0.699 ± 0.009 | 0.907 ± 0.006 |
| **Unweighted extra trees, 12 inputs with h_min_pair (final)** | **0.712 ± 0.009** | **0.913 ± 0.008** |
| Majority-class baseline | 0.259 | |

For the combination model, balanced and unweighted extra trees have the same accuracy (12 inputs: top-1 0.714 vs 0.712, top-3 0.914 vs 0.913), but the unweighted model is better calibrated: alloy-level multiclass Brier 0.409 vs 0.412, and top-1 confidence ECE 0.042 vs 0.048 at alloy level and 0.044 vs 0.061 at row level. The final combination model is therefore unweighted. Dropping the class weights also helps the random forest (top-1 0.676 to 0.698).

### Baseline: BCC / FCC / other

The first version of this project predicted three classes. Alloy-level results with a random forest (balanced) and 42 features, before adding mixing enthalpy:

| Model | Row macro F1 | Alloy accuracy | Alloy macro F1 | FCC recall | Other recall |
|---|---|---|---|---|---|
| Majority-class baseline | 0.214 ± 0.000 | 0.615 | 0.254 | 0.00 | 1.00 |
| Logistic regression | 0.686 ± 0.021 | 0.724 ± 0.007 | 0.675 ± 0.009 | 0.574 ± 0.021 | 0.777 ± 0.004 |
| **Random forest (balanced)** | **0.722 ± 0.018** | **0.786 ± 0.013** | **0.748 ± 0.014** | 0.652 ± 0.025 | 0.823 ± 0.017 |

Adding mixing enthalpy left the random forest unchanged (alloy macro F1 0.748 to 0.748) and improved logistic regression (0.675 to 0.700, alloy accuracy 0.724 to 0.749). The 3-class setup hides most of the structure, since `other` lumps together B2, Laves and secondary-phase alloys, which is why the project moved to multi-label prediction.

![Feature importance](research/feature_importance.png)

Impurity-based feature importance of the 3-class random forest, trained on the six original physics descriptors and the five processing flags. The five processing flags are summed into one bar (Process). VEC contributes the most (about 0.28), followed by mean melting point (about 0.19), size mismatch (about 0.15) and mixing enthalpy (about 0.14). Processing as a whole contributes about 0.06. These values come from one model fitted on all data and are less reliable than the permutation importances above, which are computed on held-out folds.

## Web app

`app.py` is a Streamlit app. Enter a composition and a processing route to get a probability for each of the five phases, the three most likely phase combinations, and the seven computed descriptors. It also warns when the composition is in the training data (probabilities are then optimistic), when elements are rare in the data, and that Laves and B2 predictions are the least reliable.

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
| `research/feature.py` | Parses formulas into element fractions and computes composition-based descriptors, including mixing enthalpy and the strongest pair enthalpy |
| `research/label.py` | Splits the microstructure string into binary phase labels |
| `research/modelling.py` | 3-class baseline with grouped cross-validation, and the feature importance figure for the six original physics descriptors and the processing flags |
| `research/modelling_multilabel.py` | Multi-label evaluation: per-phase average precision for the baseline models, the random forest vs extra trees and `h_min_pair` comparison, permutation importance of the physics descriptors, and phase-combination accuracy |
| `research/leakage.py` | Random vs composition-grouped cross-validation for the per-phase and combination random forests, writes `leakage_results.csv` |
| `research/calibration.py` | Reliability curves, Brier score and ECE of the per-phase forests (balanced and unweighted, with and without sigmoid and isotonic calibration), and the reliability figure |
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

## Features (44)

| Group | Count | Description |
|---|---|---|
| Element fractions | 28 | Molar fraction of each of the 28 elements in the dataset |
| Mean elemental properties | 4 | Composition-weighted mean atomic radius, electronegativity, melting point and VEC |
| Mismatch descriptors | 2 | Atomic size mismatch (delta, %) and electronegativity spread (delta chi) |
| Mixing entropy | 1 | Ideal mixing entropy, -R sum(c ln c) |
| Mixing enthalpy | 1 | Miedema model, delta H mix = sum over pairs of 4 H_ij c_i c_j |
| Strongest pair enthalpy | 1 | `h_min_pair`, the most negative Miedema H_ij among the pairs of elements present |
| Number of elements | 1 | Count of elements with non-zero fraction |
| Processing route | 5 | One-hot: cast, wrought, anneal, powder, other |
| Density | 1 | Calculated density from the source dataset |

The final model and the app use 12 inputs: seven physics descriptors (delta, delta H mix, delta S mix, VEC, mean melting point, delta chi, h_min_pair) and five processing flags. Microstructure, yield strength and hardness are not used as inputs, since microstructure is a finer description of the target and would leak the answer.

## Validation design

The dataset contains many repeated compositions: 831 of 1354 rows repeat a formula already seen, and one alloy (HfNbTaTiZr) alone accounts for 86 rows. A random split would put copies of the same alloy in both train and test and inflate every score.

- Folds are built with `StratifiedGroupKFold`, grouping by composition normalised to atomic fractions (so `Co1 Fe1 Ni1` and `Co2 Fe2 Ni2` count as one alloy). For stratification each alloy gets a single priority label (Laves, B2, Sec, FCC, otherwise BCC).
- Splits are checked with assertions: no composition appears in two folds and every row is assigned.
- Every comparison is repeated over 5 split seeds. Differences smaller than the seed-to-seed standard deviation are treated as noise.
- Feature scaling for logistic regression is fitted inside each training fold through a pipeline.

## How much does a random split inflate the scores?

`research/leakage.py` reruns the random forest models with a plain random split and compares them to the grouped split. Both use 5 folds, the same 5 seeds, the same stratification label and the same alloy-level metrics. The only difference is that the random split (`StratifiedKFold`) ignores composition, so on average 72% of test rows have the same composition in the training folds (0% for the grouped split). The gap is random minus grouped, mean ± std over the 5 seeds.

| Phase | Prevalence | RF all 44 features: grouped | random | gap | RF physics + processing: grouped | random | gap |
|---|---|---|---|---|---|---|---|
| BCC | 0.628 | 0.985 ± 0.002 | 0.992 ± 0.000 | +0.007 ± 0.002 | 0.974 ± 0.005 | 0.988 ± 0.003 | +0.014 ± 0.007 |
| FCC | 0.471 | 0.980 ± 0.003 | 0.988 ± 0.007 | +0.007 ± 0.009 | 0.975 ± 0.009 | 0.981 ± 0.009 | +0.006 ± 0.014 |
| B2 | 0.148 | 0.852 ± 0.021 | 0.897 ± 0.015 | +0.045 ± 0.012 | 0.830 ± 0.025 | 0.893 ± 0.022 | +0.063 ± 0.017 |
| Laves | 0.071 | 0.594 ± 0.038 | 0.888 ± 0.020 | **+0.294 ± 0.051** | 0.552 ± 0.036 | 0.859 ± 0.021 | **+0.307 ± 0.044** |
| Secondary phase | 0.305 | 0.859 ± 0.011 | 0.908 ± 0.009 | +0.049 ± 0.013 | 0.839 ± 0.015 | 0.898 ± 0.009 | +0.059 ± 0.016 |

With the 6 physics descriptors alone the gaps are BCC +0.013, FCC +0.010, B2 +0.097, Laves +0.389 and secondary phase +0.070 (Laves AP 0.474 grouped vs 0.863 random).

| Phase combinations | Grouped | Random | Gap |
|---|---|---|---|
| Physics, top-1 | 0.652 ± 0.009 | 0.741 ± 0.007 | +0.089 ± 0.009 |
| Physics, top-3 | 0.904 ± 0.013 | 0.934 ± 0.007 | +0.031 ± 0.013 |
| Physics + processing, top-1 | 0.676 ± 0.013 | 0.764 ± 0.004 | +0.088 ± 0.010 |
| Physics + processing, top-3 | 0.910 ± 0.013 | 0.944 ± 0.012 | +0.034 ± 0.009 |

- **Leakage is small for BCC and FCC and large for the rare phases.** BCC and FCC gain under 0.015 AP, close to seed noise, because VEC separates them well for unseen alloys too. B2 and secondary phases gain 0.045 to 0.10.
- **A random split makes Laves look solved.** Laves AP rises from about 0.5 to about 0.86 to 0.89, so most of the apparent Laves skill under a random split is memorising repeated alloys.
- **Combination accuracy is inflated by about 9 points at top-1** (0.68 to 0.76 with processing).
- **Removing element fractions does not remove the leakage.** With the 6 physics descriptors only, the random split lifts Laves by 0.39 and B2 by 0.10, more than with all 44 features, because six continuous descriptors are still enough to recognise a repeated composition.

## Limitations

- Composition and a coarse processing category are the only inputs. Heat treatment, which controls precipitation of secondary phases, is not captured.
- Laves and B2 predictions are the least reliable, and Laves AP varies the most across seeds.
- 32 formulas carry conflicting labels across records (different processing, or label noise), which caps achievable accuracy.
- Compositions that differ only slightly (for example Al0.3 and Al0.304) are treated as different alloys and can still fall on opposite sides of a split.
- 19 element pairs have no tabulated Miedema value and are set to zero when computing mixing enthalpy.
- Probabilities are not post-hoc calibrated, and the permutation importances above come from a single seed of the random forest.
- VEC values are hand-entered and follow one literature convention.

## Possible extensions

- Test whether dropping size mismatch alone changes Laves performance
- Calibrate the probabilities (the Omega parameter was tried for Laves and did not help)
- Use the same features for hardness and yield strength regression

## Reproduce

```bash
pip install pandas numpy scikit-learn pymatgen matplotlib
python research/data.py
python research/feature.py
python research/label.py
python research/modelling_multilabel.py
python research/leakage.py
python research/calibration.py
python train_final.py
streamlit run app.py
```

Run every command from the repository root. The numbers in this README were produced with scikit-learn 1.9.1, the version pinned in `requirements.txt`. Random forest scores shift slightly across scikit-learn versions.
