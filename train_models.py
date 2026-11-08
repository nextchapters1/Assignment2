"""Train, evaluate, and save classifiers for the WDBC breast cancer dataset."""

from pathlib import Path

import joblib
import pandas as pd
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
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DATA_PATH = PROJECT_ROOT / "data" / "wdbc.data"
FULL_CSV_PATH = PROJECT_ROOT / "data" / "breast_cancer_wdbc.csv"
TEST_DATA_PATH = PROJECT_ROOT / "test_data.csv"
MODEL_DIR = PROJECT_ROOT / "model"

# Each measurement is recorded as a mean, standard error, and worst value.
CORE_MEASUREMENTS = [
    "radius",
    "texture",
    "perimeter",
    "area",
    "smoothness",
    "compactness",
    "concavity",
    "concave_points",
    "symmetry",
    "fractal_dimension",
]

FEATURE_COLUMNS = (
    [f"{feat}_mean" for feat in CORE_MEASUREMENTS]
    + [f"{feat}_se" for feat in CORE_MEASUREMENTS]
    + [f"{feat}_worst" for feat in CORE_MEASUREMENTS]
)

TARGET_COLUMN = "diagnosis"

# Fixed random seeds make the train/test split and applicable models repeatable.
CLASSIFIERS = {
    "logistic_regression": LogisticRegression(max_iter=5000, random_state=42),
    "decision_tree": DecisionTreeClassifier(random_state=42),
    "knn": KNeighborsClassifier(n_neighbors=5),
    "naive_bayes": GaussianNB(),
    "svm": SVC(probability=True, random_state=42),
}


def load_and_preprocess_data() -> pd.DataFrame:
    """Name the raw dataset columns, drop the ID, and save the cleaned data."""
    column_names = ["id", TARGET_COLUMN] + FEATURE_COLUMNS
    raw_data = pd.read_csv(RAW_DATA_PATH, header=None, names=column_names)

    clean_data = raw_data.drop(columns=["id"])
    clean_data.to_csv(FULL_CSV_PATH, index=False)
    return clean_data


def compute_evaluation_metrics(
    model: Pipeline, test_features: pd.DataFrame, test_labels: pd.Series
) -> dict[str, float]:
    """Calculate classification metrics on the held-out test split."""
    predictions = model.predict(test_features)
    probabilities = model.predict_proba(test_features)[:, 1]

    return {
        "Accuracy": accuracy_score(test_labels, predictions),
        "AUC": roc_auc_score(test_labels, probabilities),
        "Precision": precision_score(test_labels, predictions, zero_division=0),
        "Recall": recall_score(test_labels, predictions, zero_division=0),
        "F1": f1_score(test_labels, predictions, zero_division=0),
        "MCC": matthews_corrcoef(test_labels, predictions),
    }


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    dataset = load_and_preprocess_data()
    print(f"Loaded {len(dataset)} rows and {len(FEATURE_COLUMNS)} features.")
    print(f"Diagnosis counts: {dict(dataset[TARGET_COLUMN].value_counts())}")

    train_data, test_data = train_test_split(
        dataset,
        test_size=0.20,
        stratify=dataset[TARGET_COLUMN],
        random_state=42,
    )

    test_data.to_csv(TEST_DATA_PATH, index=False)
    print(f"Saved {len(test_data)} held-out rows to {TEST_DATA_PATH.name}.")

    train_features = train_data[FEATURE_COLUMNS]
    train_labels = (train_data[TARGET_COLUMN] == "M").astype(int)

    test_features = test_data[FEATURE_COLUMNS]
    test_labels = (test_data[TARGET_COLUMN] == "M").astype(int)

    print(
        f"Training on {len(train_features)} rows and testing on "
        f"{len(test_features)} rows."
    )

    results = []
    for model_name, estimator in CLASSIFIERS.items():
        pipeline = make_pipeline(StandardScaler(), estimator)
        pipeline.fit(train_features, train_labels)

        artifact_path = MODEL_DIR / f"{model_name}.joblib"
        joblib.dump(pipeline, artifact_path)

        metrics = compute_evaluation_metrics(pipeline, test_features, test_labels)
        results.append({"Model": model_name, **metrics})
        print(
            f"Trained {model_name}: accuracy={metrics['Accuracy']:.4f}, "
            f"AUC={metrics['AUC']:.4f} (saved {artifact_path.name})."
        )

    results_table = pd.DataFrame(results).set_index("Model").round(4)
    metrics_path = MODEL_DIR / "metrics.csv"
    results_table.to_csv(metrics_path)

    print(f"\nSaved metrics to {metrics_path.relative_to(PROJECT_ROOT)}:\n")
    print(results_table.to_string())


if __name__ == "__main__":
    main()
