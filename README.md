# Respiratory Infection Diagnosis from Relational Medical Records

Multiclass classification of an index encounter into one of four respiratory diagnoses, using triage data and medical history spread across 15 relational tables.

**Classes:** `viral_sinusitis`, `acute_bronchitis`, `viral_pharyngitis`, `strep_pharyngitis` (minority class, about 5%)
**Metric:** macro F1
**Data size:** 5,000 index encounters (3,600 train / 1,400 test)

## Results

Cross-validated macro F1 (stratified 5-fold, repeated over 3 fold seeds, strep class weight tuned):

| Model | Macro F1 (screening) | Macro F1 (fully nested) |
|---|---|---|
| **LightGBM (tuned) + CatBoost** (final) | 0.726 | 0.720 ± 0.003 |
| LightGBM (tuned) (second submission) | 0.713 | 0.718 ± 0.000 |
| CatBoost | 0.719 | 0.713 ± 0.004 |
| Compact logistic regression (29 features) | 0.720 | n/a |

The ensemble's lead over LightGBM alone is within noise. The compact logistic regression is nearly as accurate and fully interpretable.

## Leaderboard

| Submission | Public score | Private score |
|---|---|---|
| `submission_lgbm_catboost.csv` (final ensemble) | 0.72975 | **0.76944** |
| `submission_lgbm.csv` (LightGBM only) | 0.73108 | 0.74824 |

The final ensemble placed **3rd of 61 on the private leaderboard** (17th of 61 on the public leaderboard). The model was selected by repeated, fully nested cross-validation rather than by public leaderboard score, and the jump from public to private rank is consistent with that choice.

## Approach

1. **Data quality audit**: keys, class imbalance, train/test time coverage, temporal leakage checks, missing values.
2. **Cleaning**: decimal commas, mixed units (mmHg vs kPa, SpO2 percent vs fraction), implausible vital values, county typos (fuzzy matching), explicit cutoff filter on medical history.
3. **Feature engineering**: symptom-group flags and severities, demographics, medical-history counts and code flags, and Centor-inspired clinical features for strep pharyngitis.
4. **Modeling**: feature-block ablation, algorithm comparison (LogReg, Random Forest, XGBoost, LightGBM, CatBoost), Optuna tuning, ensembling, and a strep-weight decision rule.
5. **Evaluation**: fully nested evaluation for the final candidates, error analysis.
6. **Interpretability**: gain importance, SHAP (global, per-class, local), logistic regression coefficients.
7. **Submission**: 5-seed refit on all training data, two prediction files.

## Repository structure

```
.
├── respiratory-infection-diagnosis.ipynb  
├── requirements.txt
├── data/                                   
├── submissions/                            
├── .gitignore
├── LICENSE
└── README.md
```

## Getting started

```bash
git clone https://github.com/<username>/<repo-name>.git
cd <repo-name>
pip install -r requirements.txt
```

### Data

The competition dataset is not included in this repository. Download it from the competition page and arrange it like this:

```
data/
├── train.csv
├── test.csv
├── sample_submission.csv
├── index_encounters.csv
├── patients.csv
├── Triase/Triase/
│   ├── triage_vitals.csv
│   ├── triage_symptoms.csv
│   ├── triage_symptom_severity.csv
│   └── presenting_complaints.csv
└── Riwayat Medis/Riwayat Medis/
    ├── previous_encounter.csv
    ├── conditions.csv
    ├── observations.csv
    ├── medications.csv
    ├── procedures.csv
    ├── immunizations.csv
    └── allergies.csv
```

### Running

The notebook was developed on Google Colab. To run it locally, change this line in the setup cell:

```python
DATA_DIR = "data"   # originally "/content/drive/MyDrive/gdgoc-ugm"
```

and skip the `!pip install` cell if dependencies are already installed. `RUN_TUNING = False` reuses the stored best LightGBM parameters; set it to `True` to rerun the Optuna search (slow).

Outputs: `submission_lgbm_catboost.csv` (final) and `submission_lgbm.csv` (second submission).

## Tech stack

Python, pandas, NumPy, scikit-learn, LightGBM, XGBoost, CatBoost, Optuna, SHAP, matplotlib, seaborn.

## Author

**Aqil Syafiq Alhafidz**

## License

Code is released under the MIT License (see `LICENSE`). The competition data is subject to its own terms and is not redistributed here.
