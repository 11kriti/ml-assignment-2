"""Train and evaluate five classification models for ML Assignment 2."""

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "model"


def build_models():
    return {
        "Logistic Regression": Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=3000, random_state=42)),
        ]),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
        "kNN": Pipeline([
            ("scale", StandardScaler()),
            ("model", KNeighborsClassifier(n_neighbors=7)),
        ]),
        "Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=8, random_state=42, n_jobs=-1
        ),
    }


def load_data():
    dataset = load_breast_cancer(as_frame=True)
    X = dataset.data
    y = dataset.target
    return X, y


def evaluate(model, X_test, y_test):
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    return {
        "Accuracy": accuracy_score(y_test, predictions),
        "AUC": roc_auc_score(y_test, probabilities),
        "Precision": precision_score(y_test, predictions, zero_division=0),
        "Recall": recall_score(y_test, predictions, zero_division=0),
        "F1": f1_score(y_test, predictions, zero_division=0),
        "MCC": matthews_corrcoef(y_test, predictions),
    }


def main():
    MODEL_DIR.mkdir(exist_ok=True)
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    test_data = X_test.copy()
    test_data["target"] = y_test.to_numpy()
    test_data.to_csv(ROOT / "test_data.csv", index=False)

    results = {}
    for name, model in build_models().items():
        model.fit(X_train, y_train)
        results[name] = evaluate(model, X_test, y_test)
        joblib.dump(model, MODEL_DIR / (name.lower().replace(" ", "_") + ".joblib"))

    with (MODEL_DIR / "metrics.json").open("w") as handle:
        json.dump(results, handle, indent=2)
    print(pd.DataFrame(results).T.round(4).to_string())


if __name__ == "__main__":
    main()
