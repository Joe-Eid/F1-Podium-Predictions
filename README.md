# Formula 1 Podium Prediction — Milestone 1

**Team 95**

| Name | ID |
|---|---|
| Youssef Mohamed Mahmoud Eid | 52-2413 |
| Omar Khaled Mohamed Mohamed Ghonima | 58-1891 |
| Mohamed Ayman Mohamed Abdelrahman Elsayed | 58-22010 |
| Omar Ahmed Mohamed Mohamed Mohamed Elbehery | 58-16880 |

Predict whether a driver finishes on the podium (`positionOrder <= 3`) using only information available **after qualifying and before the race starts**.

- **Dataset:** [Formula 1 World Championship (1950–2024)](https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020) (Kaggle / Ergast, 14 relational CSV tables)
- **Notebook:** `F1_Podium_Prediction.ipynb` — a single run-all notebook containing the audit, cleaning, data-engineering answers, feature engineering, models, experiments, XAI and the inference function
- **Report:** `report/F1_Podium_Report.pdf`
- **Figures:** `figures/` (saved by the notebook)

## How to run
- **On Kaggle:** add the dataset `rohanrao/formula-1-world-championship-1950-2020` to the notebook and click **Run All**. The CSV folder is found automatically under `/kaggle/input`.
- **Locally:** run `pip install pandas numpy matplotlib seaborn scikit-learn keras tensorflow shap lime kagglehub`, then run the notebook. The dataset is downloaded with `kagglehub`. Alternatively, set `F1_DATA_DIR` to a folder that contains the CSVs.

## Notebook sections
1. Data loading (raw tables are kept immutable; cleaning works on copies)
2. EDA before cleaning
3. Cleaning & auditing: `\N` sentinels, semantic typing (dates, durations → seconds), primary/foreign keys, duplicate `(raceId, driverId)` rule
4. EDA after cleaning (impact of the cleaning)
5. Data-engineering questions: front-row conversion by circuit (pre/post 2014), home advantage controlled for grid, mechanical retirement rates by constructor
6. Leakage-safe features in 4 groups: grid, qualifying, form (previous 5 races), standings (before the race)
7. Time-based split: train 2003–2017, validation 2018–2021, test 2022–2024
8. Pre-processing (median imputation, scaling) and two pre-processing-order experiments
9. Models: grid-rule reference, Logistic Regression (baseline), Random Forest, shallow Keras NN
10. Model comparison (ROC-AUC, PR-AUC, F1, Precision, Recall on train / validation / test)
11. Feature-group ablation
12. XAI: permutation importance, SHAP (global and local), LIME
13. Inference function `predict_podium(record)`, demonstrated on complete and partially-missing records
14. Conclusions and limitations

## Main results (unseen test seasons 2022–2024)
| Model | ROC-AUC | PR-AUC | F1 |
|---|---|---|---|
| Rule: grid position | 0.914 | 0.647 | 0.649 |
| Logistic Regression (baseline) | 0.935 | 0.700 | 0.667 |
| **Random Forest (final)** | **0.940** | **0.708** | **0.668** |
| Shallow NN | ~0.93 | ~0.69 | ~0.67 |

The NN numbers vary slightly between runs and backends (TensorFlow vs. PyTorch).
