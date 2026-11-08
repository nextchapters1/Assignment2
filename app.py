from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
DEFAULT_TEST_FILE = ROOT / "test_data.csv"
TARGET = "diagnosis"

MODEL_LABELS = {
    "logistic_regression": "Logistic Regression",
    "decision_tree": "Decision Tree",
    "knn": "K-Nearest Neighbors",
    "naive_bayes": "Gaussian Naive Bayes",
    "svm": "Support Vector Machine",
}


@st.cache_resource
def load_models() -> dict[str, Pipeline]:
    """Load the trained classifiers once and reuse them across app reruns."""
    return {key: joblib.load(MODEL_DIR / f"{key}.joblib") for key in MODEL_LABELS}


def get_features(models: dict[str, Pipeline]) -> list[str]:
    """Return the input columns expected by the saved model pipelines."""
    first = next(iter(models.values()))
    return list(first.feature_names_in_)


def compute_metrics(
    model: Pipeline, features: pd.DataFrame, true_labels: pd.Series
) -> dict[str, float]:
    """Calculate the evaluation metrics displayed for a model."""
    predictions = model.predict(features)
    probabilities = model.predict_proba(features)[:, 1]
    return {
        "Accuracy": accuracy_score(true_labels, predictions),
        "AUC": roc_auc_score(true_labels, probabilities),
        "Precision": precision_score(true_labels, predictions, zero_division=0),
        "Recall": recall_score(true_labels, predictions, zero_division=0),
        "F1": f1_score(true_labels, predictions, zero_division=0),
        "MCC": matthews_corrcoef(true_labels, predictions),
    }


st.set_page_config(page_title="Breast Cancer Classifier", layout="wide")
st.title("Breast Cancer Diagnosis: Classifier Evaluation")
st.write(
    "Upload a test CSV containing the 30 WDBC feature columns and the "
    "`diagnosis` column (M = malignant, B = benign). Saved models are "
    "evaluated on the uploaded data; nothing is retrained."
)

models = load_models()
features = get_features(models)

with st.sidebar:
    st.header("Input")
    uploaded = st.file_uploader("Upload test CSV", type=["csv"])
    model_key = st.selectbox(
        "Select model",
        options=list(MODEL_LABELS),
        format_func=lambda k: MODEL_LABELS[k],
    )

if uploaded is not None:
    data = pd.read_csv(uploaded)
    source = uploaded.name
else:
    data = pd.read_csv(DEFAULT_TEST_FILE)
    source = "test_data.csv (bundled)"

st.caption(f"Data source: {source} | Rows: {len(data)}")

missing = [c for c in features + [TARGET] if c not in data.columns]
if missing:
    st.error(f"Uploaded CSV is missing required columns: {', '.join(missing)}")
    st.stop()

feature_data = data[features]
true_labels = (data[TARGET].astype(str).str.strip().str.upper() == "M").astype(int)

if true_labels.nunique() < 2:
    st.warning("The uploaded data contains only one class; AUC and MCC are undefined.")

model = models[model_key]
metrics = compute_metrics(model, feature_data, true_labels)

st.subheader(f"Evaluation metrics: {MODEL_LABELS[model_key]}")
cols = st.columns(6)
for col, (name, value) in zip(cols, metrics.items()):
    col.metric(name, f"{value:.4f}")

left, right = st.columns(2)
with left:
    st.subheader("Confusion matrix")
    predictions = model.predict(feature_data)
    cm = confusion_matrix(true_labels, predictions, labels=[0, 1])
    cm_df = pd.DataFrame(
        cm,
        index=["Actual Benign (0)", "Actual Malignant (1)"],
        columns=["Predicted Benign (0)", "Predicted Malignant (1)"],
    )
    st.dataframe(cm_df)

with right:
    st.subheader("Classification report")
    report = classification_report(
        true_labels,
        predictions,
        target_names=["Benign", "Malignant"],
        output_dict=True,
        zero_division=0,
    )
    st.dataframe(pd.DataFrame(report).T.round(4))

st.subheader("All models on the uploaded data")
comparison = pd.DataFrame(
    {
        MODEL_LABELS[key]: compute_metrics(model, feature_data, true_labels)
        for key, model in models.items()
    }
).T
st.dataframe(comparison.round(4))
