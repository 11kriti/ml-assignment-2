"""Interactive Streamlit application for ML Assignment 2."""

from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import classification_report, confusion_matrix


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "kNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Random Forest": "random_forest.joblib",
}


@st.cache_resource
def load_models():
    return {name: joblib.load(MODEL_DIR / filename) for name, filename in MODEL_FILES.items()}


@st.cache_data
def dataset_schema():
    data = load_breast_cancer(as_frame=True)
    return list(data.feature_names), data.target_names.tolist()


def main():
    st.set_page_config(page_title="Breast Cancer Classifier", page_icon="🧬", layout="wide")
    st.title("🧬 Breast Cancer Classification Lab")
    st.caption("ML Assignment 2 | Five supervised classification models evaluated on one dataset")

    feature_names, target_names = dataset_schema()
    models = load_models()
    uploaded = st.file_uploader(
        "Optional: upload another test CSV",
        type="csv",
        help="If no file is selected, the bundled test_data.csv is used automatically.",
    )
    if uploaded is None:
        data = pd.read_csv(ROOT / "test_data.csv")
        st.success("Using the bundled test_data.csv file.")
    else:
        data = pd.read_csv(uploaded)
    missing = sorted(set(feature_names) - set(data.columns))
    if missing:
        st.error(f"The file is missing {len(missing)} required feature column(s).")
        st.write(missing)
        return

    X = data[feature_names]
    has_target = "target" in data.columns
    selected = st.selectbox("Choose a model", list(models))
    model = models[selected]
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]

    if has_target:
        from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score
        y = data["target"]
        metrics = {
            "Accuracy": accuracy_score(y, predictions),
            "AUC": roc_auc_score(y, probabilities),
            "Precision": precision_score(y, predictions, zero_division=0),
            "Recall": recall_score(y, predictions, zero_division=0),
            "F1": f1_score(y, predictions, zero_division=0),
            "MCC": matthews_corrcoef(y, predictions),
        }
        cols = st.columns(6)
        for col, (label, value) in zip(cols, metrics.items()):
            col.metric(label, f"{value:.3f}")

        left, right = st.columns(2)
        with left:
            st.subheader("Confusion matrix")
            fig, ax = plt.subplots(figsize=(4.5, 3.5))
            sns.heatmap(confusion_matrix(y, predictions), annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
            ax.set_xlabel("Predicted label")
            ax.set_ylabel("True label")
            st.pyplot(fig, clear_figure=True)
        with right:
            st.subheader("Classification report")
            report = classification_report(y, predictions, target_names=target_names, output_dict=True, zero_division=0)
            st.dataframe(pd.DataFrame(report).T.round(3), use_container_width=True)
    else:
        st.warning("No target column found, so predictions are shown without evaluation metrics.")

    output = data.copy()
    output["prediction"] = predictions
    output["prediction_label"] = [target_names[int(value)] for value in predictions]
    st.subheader("Predictions")
    st.dataframe(output.head(100), use_container_width=True)
    st.download_button("Download predictions", output.to_csv(index=False), "predictions.csv", "text/csv")


if __name__ == "__main__":
    main()
