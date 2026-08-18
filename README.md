# ML Assignment 2: Breast Cancer Classification

## Problem statement

Build, evaluate, and demonstrate multiple classification models on one public classification dataset. The solution must expose the trained models through an interactive Streamlit application that accepts test data, displays evaluation metrics, and shows a confusion matrix and classification report.

## Dataset description

This project uses the **Breast Cancer Wisconsin (Diagnostic)** dataset, distributed through scikit-learn and originally published by the UCI Machine Learning Repository. It contains 569 instances, 30 real-valued features, and a binary target: malignant (0) or benign (1). The dataset exceeds the assignment minimum of 500 instances and 12 features. The included `test_data.csv` contains the stratified 20% hold-out test split (114 rows) generated with `random_state=42`.

Source: https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic

## GitHub Repository Link

**Add your repository URL here after publishing:** `https://github.com/<your-username>/<your-repository>`

## Live Streamlit App Link

**Add your deployed app URL here after deployment:** `https://<your-app-name>.streamlit.app`

## Models and evaluation

All models use the same train/test split. Logistic Regression and kNN use standardization; the tree-based models use the original feature values.

| ML Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9825 | 0.9954 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |
| Decision Tree | 0.9211 | 0.9163 | 0.9565 | 0.9167 | 0.9362 | 0.8341 |
| kNN | 0.9737 | 0.9884 | 0.9600 | 1.0000 | 0.9796 | 0.9442 |
| Naive Bayes | 0.9386 | 0.9878 | 0.9452 | 0.9583 | 0.9517 | 0.8676 |
| Random Forest (Ensemble) | 0.9474 | 0.9937 | 0.9583 | 0.9583 | 0.9583 | 0.8869 |

## Observations

| ML Model | Observation about model performance |
|---|---|
| Logistic Regression | Best overall balance in this run. Standardization and the approximately separable feature space produce very strong discrimination and the highest accuracy. |
| Decision Tree | Easiest to explain, but the single tree is less stable and less accurate than the other models; its depth limit reduces overfitting at the cost of some recall/precision. |
| kNN | Performs strongly after scaling because distance comparisons become meaningful across differently scaled features. It is slightly below Logistic Regression and requires storing the training set at prediction time. |
| Naive Bayes | Fast and competitive. Its conditional-independence assumption is imperfect for correlated medical measurements, but it still achieves a high AUC and strong classification scores. |
| Random Forest (Ensemble) | More robust than one tree and has excellent AUC, but this configuration does not surpass Logistic Regression on the fixed hold-out split. |
| **Overall winner** | **Logistic Regression**, based on the highest accuracy, F1, and MCC in this reproducible run. |

## Streamlit features

- CSV test-data upload with schema validation.
- Model-selection dropdown for all five models.
- Accuracy, AUC, precision, recall, F1, and MCC display when the target column is present.
- Confusion matrix, classification report, predictions table, and prediction download.

## Repository structure

```text
ml_assignment_2/
├── app.py
├── requirements.txt
├── README.md
├── test_data.csv
└── model/
    ├── model_training.py
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    ├── random_forest.joblib
    └── metrics.json
```

## Run locally

```bash
pip install -r requirements.txt
python model/model_training.py
streamlit run app.py
```

Open the local URL, upload `test_data.csv`, and select a model.

