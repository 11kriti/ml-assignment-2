"""
Streamlit demo app for the Adult Census Income classification models.

Upload the test_data.csv (or any CSV with the same 14 feature columns,
optionally including the true 'income' label), pick a model, and see
predictions plus evaluation metrics / confusion matrix.
"""

import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "kNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Random Forest (Ensemble)": "random_forest.joblib",
}

FEATURE_COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "sex",
    "capital_gain", "capital_loss", "hours_per_week", "native_country",
]
TARGET = "income"

st.set_page_config(page_title="Adult Income Classifier", layout="wide")


@st.cache_resource
def load_model(filename):
    return joblib.load(os.path.join(MODEL_DIR, filename))


@st.cache_resource
def load_label_encoder():
    return joblib.load(os.path.join(MODEL_DIR, "label_encoder.joblib"))


st.title("Adult Census Income — Classification Demo")
st.caption(
    "Dataset: UCI Adult / Census Income · Task: predict whether income is >50K or <=50K"
)

st.sidebar.header("1. Upload test data")
uploaded_file = st.sidebar.file_uploader(
    "Upload a CSV (e.g. test_data.csv)", type=["csv"]
)
st.sidebar.caption(
    "Expected columns: " + ", ".join(FEATURE_COLUMNS) + f", optionally `{TARGET}`."
)

st.sidebar.header("2. Choose a model")
model_choice = st.sidebar.selectbox("Model", list(MODEL_FILES.keys()))

if uploaded_file is None:
    st.info("Upload a CSV from the sidebar to get started. `test_data.csv` in this "
            "repo is a ready-made sample held out from training.")
    st.stop()

df = pd.read_csv(uploaded_file)
missing_cols = [c for c in FEATURE_COLUMNS if c not in df.columns]
if missing_cols:
    st.error(f"Uploaded CSV is missing required columns: {missing_cols}")
    st.stop()

st.subheader("Preview of uploaded data")
st.dataframe(df.head(10), use_container_width=True)

model = load_model(MODEL_FILES[model_choice])
label_encoder = load_label_encoder()

X = df[FEATURE_COLUMNS]
predictions = model.predict(X)
probabilities = model.predict_proba(X)[:, 1]

results = df.copy()
results["predicted_income"] = label_encoder.inverse_transform(predictions)
results["probability_>50K"] = np.round(probabilities, 4)

st.subheader(f"Predictions — {model_choice}")
st.dataframe(results.head(20), use_container_width=True)

has_ground_truth = TARGET in df.columns

if has_ground_truth:
    y_true = label_encoder.transform(df[TARGET])

    accuracy = accuracy_score(y_true, predictions)
    auc = roc_auc_score(y_true, probabilities)
    precision = precision_score(y_true, predictions)
    recall = recall_score(y_true, predictions)
    f1 = f1_score(y_true, predictions)
    mcc = matthews_corrcoef(y_true, predictions)

    st.subheader("Evaluation metrics")
    metric_cols = st.columns(6)
    metric_cols[0].metric("Accuracy", f"{accuracy:.4f}")
    metric_cols[1].metric("AUC", f"{auc:.4f}")
    metric_cols[2].metric("Precision", f"{precision:.4f}")
    metric_cols[3].metric("Recall", f"{recall:.4f}")
    metric_cols[4].metric("F1 Score", f"{f1:.4f}")
    metric_cols[5].metric("MCC", f"{mcc:.4f}")

    left, right = st.columns(2)

    with left:
        st.subheader("Confusion matrix")
        cm = confusion_matrix(y_true, predictions)
        fig, ax = plt.subplots(figsize=(4, 3.5))
        ConfusionMatrixDisplay(
            cm, display_labels=label_encoder.classes_
        ).plot(ax=ax, cmap="Blues", colorbar=False)
        st.pyplot(fig)

    with right:
        st.subheader("Classification report")
        report = classification_report(
            y_true, predictions, target_names=label_encoder.classes_, output_dict=True
        )
        st.dataframe(pd.DataFrame(report).transpose().round(3), use_container_width=True)
else:
    st.warning(
        f"No `{TARGET}` column found in the uploaded CSV — showing predictions only. "
        "Include the true label column to see evaluation metrics and the confusion matrix."
    )

st.divider()
st.subheader("All models — comparison on this uploaded data")
if has_ground_truth:
    comparison_rows = []
    for name, filename in MODEL_FILES.items():
        m = load_model(filename)
        preds = m.predict(X)
        proba = m.predict_proba(X)[:, 1]
        comparison_rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_true, preds),
            "AUC": roc_auc_score(y_true, proba),
            "Precision": precision_score(y_true, preds),
            "Recall": recall_score(y_true, preds),
            "F1": f1_score(y_true, preds),
            "MCC": matthews_corrcoef(y_true, preds),
        })
    comparison_df = pd.DataFrame(comparison_rows).set_index("Model").round(4)
    st.dataframe(comparison_df, use_container_width=True)
    st.bar_chart(comparison_df[["Accuracy", "AUC", "F1"]])
else:
    st.caption("Upload data with the true `income` column to compare all 5 models here.")
