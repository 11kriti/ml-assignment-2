"""
Trains 5 classifiers on the UCI Adult (Census Income) dataset and saves:
  - one fitted sklearn Pipeline per model -> model/<name>.joblib
  - metrics for all models -> model/metrics.json
  - a held-out evaluation split -> test_data.csv (used by the Streamlit app)

Run from the project root:
    source venv/bin/activate
    python model/train_models.py
"""

import json
import os

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_PATH = os.path.join(BASE_DIR, "data", "adult.data")
TEST_DATA_PATH = os.path.join(BASE_DIR, "test_data.csv")
METRICS_PATH = os.path.join(os.path.dirname(__file__), "metrics.json")

COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "sex",
    "capital_gain", "capital_loss", "hours_per_week", "native_country",
    "income",
]
NUMERIC_FEATURES = [
    "age", "fnlwgt", "education_num", "capital_gain",
    "capital_loss", "hours_per_week",
]
CATEGORICAL_FEATURES = [
    "workclass", "education", "marital_status", "occupation",
    "relationship", "race", "sex", "native_country",
]
TARGET = "income"


def load_data():
    df = pd.read_csv(RAW_PATH, header=None, names=COLUMNS, skipinitialspace=True)
    df = df.replace("?", pd.NA)
    df = df.dropna().reset_index(drop=True)
    df[TARGET] = df[TARGET].str.rstrip(".")
    return df


def build_preprocessor():
    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", numeric_pipe, NUMERIC_FEATURES),
        ("cat", categorical_pipe, CATEGORICAL_FEATURES),
    ])


MODEL_DEFS = {
    "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
    "decision_tree": DecisionTreeClassifier(max_depth=10, random_state=42),
    "knn": KNeighborsClassifier(n_neighbors=15),
    "naive_bayes": GaussianNB(),
    "random_forest": RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42),
}

DISPLAY_NAMES = {
    "logistic_regression": "Logistic Regression",
    "decision_tree": "Decision Tree",
    "knn": "kNN",
    "naive_bayes": "Naive Bayes",
    "random_forest": "Random Forest (Ensemble)",
}


def evaluate(y_true, y_pred, y_proba):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "auc": roc_auc_score(y_true, y_proba),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }


def main():
    df = load_data()
    print(f"Loaded {len(df)} rows after cleaning, {len(COLUMNS) - 1} features.")

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df[TARGET])  # >50K -> 1, <=50K -> 0
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    metrics = {}
    os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)

    for key, estimator in MODEL_DEFS.items():
        pipeline = Pipeline([
            ("preprocess", build_preprocessor()),
            ("model", estimator),
        ])
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]

        metrics[key] = {
            "display_name": DISPLAY_NAMES[key],
            **evaluate(y_test, y_pred, y_proba),
        }

        model_path = os.path.join(os.path.dirname(__file__), f"{key}.joblib")
        joblib.dump(pipeline, model_path)
        print(f"[{DISPLAY_NAMES[key]}] accuracy={metrics[key]['accuracy']:.4f} "
              f"auc={metrics[key]['auc']:.4f} mcc={metrics[key]['mcc']:.4f} -> saved {model_path}")

    joblib.dump(label_encoder, os.path.join(os.path.dirname(__file__), "label_encoder.joblib"))

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {METRICS_PATH}")

    # Held-out sample for the Streamlit app / repo (keeps the CSV small per assignment note).
    test_export = X_test.copy()
    test_export[TARGET] = label_encoder.inverse_transform(y_test)
    test_export = test_export.sample(n=min(1000, len(test_export)), random_state=42).reset_index(drop=True)
    test_export.to_csv(TEST_DATA_PATH, index=False)
    print(f"Test data ({len(test_export)} rows) saved to {TEST_DATA_PATH}")


if __name__ == "__main__":
    main()
