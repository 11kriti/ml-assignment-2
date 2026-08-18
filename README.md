# Adult Census Income — Classification Models & Streamlit App

## a. Problem Statement

Predict whether an individual's annual income exceeds **$50,000** based on
demographic and employment attributes collected in the 1994 US Census
database. This is a **binary classification** problem (`<=50K` vs `>50K`).

## b. Dataset Description

- **Source:** UCI Machine Learning Repository — [Adult / Census Income Data
  Set](https://archive.ics.uci.edu/dataset/2/adult)
- **Instances used:** 30,162 (after dropping rows with missing `?` values
  from the original 32,561)
- **Features:** 14 (6 numeric + 8 categorical)
  - Numeric: `age`, `fnlwgt`, `education_num`, `capital_gain`,
    `capital_loss`, `hours_per_week`
  - Categorical: `workclass`, `education`, `marital_status`, `occupation`,
    `relationship`, `race`, `sex`, `native_country`
- **Target:** `income` — `<=50K` or `>50K`
- **Class balance:** ~75% `<=50K`, ~25% `>50K` (moderately imbalanced)
- **Preprocessing:** median imputation + standard scaling for numeric
  features, most-frequent imputation + one-hot encoding for categorical
  features, all wrapped in a single `sklearn.Pipeline` per model so the
  exact same transform is reused at inference time in the Streamlit app.

## c. GitHub Repository Link

> _Fill in after pushing: `https://github.com/<your-username>/<your-repo>`_

## d. Models Used

Trained with an 80/20 stratified train/test split (`random_state=42`) —
see `model/train_models.py`.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8475 | 0.9022 | 0.7354 | 0.6052 | 0.6640 | 0.5711 |
| Decision Tree | 0.8530 | 0.8959 | 0.7681 | 0.5866 | 0.6652 | 0.5817 |
| kNN | 0.8384 | 0.8911 | 0.7007 | 0.6125 | 0.6536 | 0.5510 |
| Naive Bayes | 0.5826 | 0.8018 | 0.3678 | 0.9414 | 0.5290 | 0.3643 |
| Random Forest (Ensemble) | 0.8556 | 0.9146 | 0.7850 | 0.5786 | 0.6662 | 0.5877 |

_(Regenerate this table any time by re-running `python model/train_models.py`
and copying values from `model/metrics.json`.)_

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Strong, well-balanced baseline (AUC 0.90) — the decision boundary is close to linear in the encoded feature space, so a linear model captures most of the signal cheaply. |
| Decision Tree | Slightly better accuracy/MCC than logistic regression but marginally lower AUC — a single tree overfits some splits on high-cardinality categoricals (e.g. `native_country`), trading probability calibration for a few sharper decision boundaries. |
| kNN | Competitive but the weakest of the non-NB models — distance in a ~100-dimensional one-hot + scaled numeric space is less meaningful ("curse of dimensionality"), so neighbors are noisier than for the other models. |
| Naive Bayes | Clearly the weakest on accuracy/precision/MCC, but by far the highest recall (0.94). GaussianNB assumes feature independence and Gaussian-distributed inputs; one-hot encoded categoricals violate both assumptions badly, so it over-predicts the minority `>50K` class. |
| Random Forest (Ensemble) | Best overall — highest accuracy, AUC, and MCC. Averaging many decision trees reduces the overfitting/variance seen in the single Decision Tree while still modeling non-linear interactions between features that Logistic Regression cannot. |
| **Overall Winner for your dataset?** | **Random Forest (Ensemble)** — best accuracy (0.8556), AUC (0.9146), and MCC (0.5877), making it the most reliable model across both threshold-based and probability-based metrics. |

## Project Structure

```
project-folder/
│-- app.py                     # Streamlit app
│-- requirements.txt
│-- README.md
│-- test_data.csv              # held-out sample used for demo/evaluation in the app
│-- data/
│   └── adult.data             # full raw dataset (source for training)
│-- model/
│   ├── train_models.py        # trains all 5 models + saves metrics
│   ├── metrics.json           # metrics for all models (source for the table above)
│   ├── label_encoder.joblib
│   ├── logistic_regression.joblib
│   ├── decision_tree.joblib
│   ├── knn.joblib
│   ├── naive_bayes.joblib
│   └── random_forest.joblib
```

## Running Locally

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

# (optional) retrain models from scratch
python model/train_models.py

streamlit run app.py
```

## Streamlit App Features

- CSV upload (sidebar) — accepts `test_data.csv` or any CSV with the same
  14 feature columns, with or without the true `income` label
- Model selection dropdown across all 5 trained models
- Evaluation metrics (Accuracy, AUC, Precision, Recall, F1, MCC) computed
  live on the uploaded data when the true label is present
- Confusion matrix + classification report
- Side-by-side comparison table/chart of all 5 models on the uploaded data

## Live Streamlit App Link

> _Fill in after deploying: `https://<your-app-name>.streamlit.app`_
