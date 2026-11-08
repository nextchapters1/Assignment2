# Breast Cancer Wisconsin (Diagnostic) Classification & Interactive Streamlit Dashboard

## a. Problem statement

Breast cancer is one of the most frequently diagnosed cancers among women worldwide, and early, accurate detection is critical to improving survival rates and clinical outcomes. In diagnostic pathology, Fine Needle Aspirate (FNA) biopsies are commonly performed on suspicious breast masses. High-resolution digitized images of these aspirates reveal geometric and morphological properties of individual cell nuclei.

The primary objective of this project is to develop and evaluate five supervised machine learning classification models to determine whether an observed breast mass is **Malignant (M)** or **Benign (B)** based on 30 continuous nuclear characteristics:

1. **Logistic Regression** (L2 regularized baseline)
2. **Decision Tree** (CART)
3. **K-Nearest Neighbors** ($k=5$)
4. **Gaussian Naive Bayes**
5. **Support Vector Machine** (RBF Kernel)

To prevent data leakage, all models are built using scikit-learn pipelines with feature standardization fitted strictly on training data. The models are evaluated on a held-out test split across six standard evaluation metrics: **Accuracy, Area Under the ROC Curve (AUC), Precision, Recall, F1-Score, and Matthews Correlation Coefficient (MCC)**. In addition, an interactive Streamlit web application is developed to load these pre-trained models, evaluate them on uploaded CSV data or the bundled test set, display confusion matrices and classification reports, and present a side-by-side comparative analysis.

---

## b. Dataset description

The analysis is conducted on the well-known **Breast Cancer Wisconsin (Diagnostic) Dataset (WDBC)** from the UCI Machine Learning Repository:

- **Source & Provenance:** UCI Machine Learning Repository ([Dataset Link](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)). Created by Dr. William H. Wolberg, W. Nick Street, and Olvi L. Mangasarian, University of Wisconsin.
- **Dataset Size:** 569 total patient records (comfortably meeting the assignment threshold of at least 500 instances).
- **Feature Space:** 30 continuous, real-valued input features (exceeding the requirement of 12+ features). The features describe 10 morphological properties of cell nuclei:

  1. *Radius* (mean of distances from center to points on the perimeter)
  2. *Texture* (standard deviation of gray-scale values)
  3. *Perimeter*
  4. *Area*
  5. *Smoothness* (local variation in radius lengths)
  6. *Compactness* ($\frac{\text{perimeter}^2}{\text{area}} - 1.0$)
  7. *Concavity* (severity of concave portions of the contour)
  8. *Concave points* (number of concave portions of the contour)
  9. *Symmetry*
  10. *Fractal dimension* ("coastline approximation" $- 1$)

  For each cell sample, three statistics are computed for each of the 10 traits:

  - **Mean** (e.g., `radius_mean`, `texture_mean`)
  - **Standard Error** (e.g., `radius_se`, `texture_se`)
  - **Worst / Largest** (mean of the three largest values, e.g., `radius_worst`, `texture_worst`)
    This results in $10 \times 3 = 30$ predictive numeric attributes.
- **Target Variable:** `diagnosis`

  - **Malignant (M):** Positive class, encoded as `1` (212 samples, 37.26%)
  - **Benign (B):** Negative class, encoded as `0` (357 samples, 62.74%)
- **Data Partitioning:**

  - An 80/20 train/test stratified split was applied using a fixed random seed (`random_state=42`) to preserve the ~63:37 benign-to-malignant class ratio in both subsets.
  - **Training Set:** 455 samples (used exclusively for pipeline fitting).
  - **Test Set:** 114 samples (saved to `test_data.csv` for independent evaluation).
- **Data Preprocessing & Leakage Safeguards:**

  - The sample identification column (`id`) was removed immediately to prevent arbitrary identifier leakage into the models.
  - The dataset has no missing or null values.
  - Feature distributions vary across several orders of magnitude (e.g., `area_worst` ranges up to ~2500, whereas `smoothness_mean` is ~0.1). To ensure distance- and gradient-based models converge appropriately, `StandardScaler` is wrapped inside each model's scikit-learn `Pipeline`. The scaler learns means and variances only on the training set and transforms test data during evaluation, preventing lookahead bias.

---

## c. GitHub Repository Link

- **GitHub Repository:** https://github.com/nextchapters1/Assignment2
- **Live Streamlit App:** https://2026ak05379-ml-assignment-2.streamlit.app/
### Project Structure

```text
├── app.py                  # Interactive Streamlit evaluation dashboard
├── train_models.py         # Data processing, pipeline training, evaluation & artifact export
├── requirements.txt        # Pinned Python package dependencies
├── README.md               # Project documentation and performance report
├── test_data.csv           # 114-row held-out test split (features + diagnosis)
├── data/
│   ├── wdbc.data           # Original UCI WDBC raw datafile
│   └── breast_cancer_wdbc.csv  # Preprocessed full dataset with header names
└── model/
    ├── logistic_regression.joblib  # Trained scikit-learn pipeline (scaler + model)
    ├── decision_tree.joblib        # Trained scikit-learn pipeline (scaler + model)
    ├── knn.joblib                  # Trained scikit-learn pipeline (scaler + model)
    ├── naive_bayes.joblib          # Trained scikit-learn pipeline (scaler + model)
    ├── svm.joblib                  # Trained scikit-learn pipeline (scaler + model)
    └── metrics.csv                 # Evaluation metrics on the held-out test set
```

---

## d. Models used

All five classification models were trained on the identical 455 training rows and evaluated on the same 114 held-out test instances (72 Benign, 42 Malignant). Malignant (`M = 1`) represents the positive class.

### 1. Quantitative Performance Comparison

| ML Model Name                          | Accuracy | AUC    | Precision | Recall | F1     | MCC    |
| -------------------------------------- | -------- | ------ | --------- | ------ | ------ | ------ |
| **Logistic Regression**          | 0.9649   | 0.9960 | 0.9750    | 0.9286 | 0.9512 | 0.9245 |
| **Decision Tree (CART)**         | 0.9298   | 0.9246 | 0.9048    | 0.9048 | 0.9048 | 0.8492 |
| **KNN ($k=5$)**                | 0.9561   | 0.9823 | 0.9744    | 0.9048 | 0.9383 | 0.9058 |
| **Naive Bayes (Gaussian)**       | 0.9211   | 0.9891 | 0.9231    | 0.8571 | 0.8889 | 0.8292 |
| **Support Vector Machine (RBF)** | 0.9737   | 0.9947 | 1.0000    | 0.9286 | 0.9630 | 0.9442 |

---

### 2. Performance Observations

| ML Model Name                    | Observation about model performance                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| -------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Logistic Regression**    | Performed exceptionally well across all evaluation criteria (Accuracy: 96.49%, AUC: 0.9960, MCC: 0.9245). Because many nuclear features (e.g., perimeter, area, concavity) have strong monotonic relationships with malignancy, a regularized linear hyperplane in standardized feature space separates the classes very cleanly. It registered the highest AUC (0.9960) of all five models, demonstrating superior probability ranking.                                                                                                         |
| **Decision Tree**          | Lowest overall performer among the models tested (Accuracy: 92.98%, MCC: 0.8492). An unpruned decision tree produces orthogonal, axis-aligned splits that overfit training noise and small local variations. With 30 continuous and collinear features, isolated decision splits lack the generalization capability of smooth decision boundaries.                                                                                                                                                                                               |
| **KNN ($k=5$)**          | Strong predictive performance (Accuracy: 95.61%, AUC: 0.9823) with high precision (0.9744). The pipeline's z-score standardization was crucial here, as unscaled variance in attributes like `area_worst` would have completely dominated Euclidean distance calculations. However, its Recall of 90.48% resulted in 4 false negatives, which is a clinical concern in tumor diagnosis.                                                                                                                                                        |
| **Naive Bayes**            | While it scored an impressive AUC of 0.9891 (indicating good relative ordering of probabilities), it had the lowest test Accuracy (92.11%) and lowest Recall (85.71%, missing 6 malignant cases). This directly highlights the breakdown of Naive Bayes' conditional independence assumption: multiple morphological attributes are geometrically dependent (e.g., radius, perimeter, and area are intrinsically linked by $P \approx 2\pi r$ and $A \approx \pi r^2$), leading to overconfident and skewed posterior probability estimates. |
| **Support Vector Machine** | Top-performing model overall, achieving the highest Accuracy (97.37%), F1-Score (0.9630), and MCC (0.9442). Utilizing the Radial Basis Function (RBF) kernel, SVM effectively mapped the 30-dimensional standardized features into a higher-dimensional space where benign and malignant nuclei were cleanly partitioned. Notably, it achieved a perfect Precision of 1.0000 (0 false positives on the test set) while maintaining a strong Recall of 0.9286 (only 3 false negatives).                                                           |

---

### 3. Model Selection & Clinical Discussion

- **Overall Winner:** **Support Vector Machine (RBF Kernel)** is the clear empirical winner on this dataset. It achieves the top scores in Accuracy (0.9737), Precision (1.0000), F1-Score (0.9630), and Matthews Correlation Coefficient (0.9442).
- **Clinical Trade-offs:** In cancer diagnostics, a **False Negative** (diagnosing a malignant tumor as benign) carries grave consequences, delaying vital clinical treatment. Conversely, a **False Positive** causes patient anxiety and prompts additional follow-up imaging or biopsies, but does not risk unmanaged disease progression.
  - Both SVM and Logistic Regression achieved the highest Recall (0.9286), missing only 3 out of 42 malignant cases in the test set.
  - However, SVM delivered **zero false positives** (Precision = 1.0000), making it the most balanced and reliable classifier overall.
  - **Logistic Regression** remains an excellent clinical runner-up: it achieved the highest AUC (0.9960), tied SVM for top recall, and offers direct feature interpretability through odds ratios—a key consideration in medical decision-support systems where clinicians value explainability over black-box predictions.

---

## How to Run Locally

### 1. Prerequisites & Installation

Clone the repository and install the dependencies in a Python 3.10+ virtual environment:

```bash
# Clone the repository
git clone https://github.com/<your-username>/Assignment2.git
cd <repo-name>

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

### 2. Train Models & Export Artifacts (Optional)

The pre-trained models and test dataset are already checked into `model/` and `test_data.csv`. To retrain from scratch:

```bash
python train_models.py
```

### 3. Launch the Streamlit App

```bash
streamlit run app.py
```

Open your browser to `http://localhost:8501`.

---

## Streamlit Application Features

- **Interactive File Upload:** Upload any custom test CSV matching the 30 WDBC features and the `diagnosis` column. If no file is uploaded, the app automatically loads the bundled 114-row held-out test split.
- **Model Selector:** Seamlessly switch between the five saved classification pipelines.
- **Key Diagnostic Metrics:** Displays real-time cards for Accuracy, ROC-AUC, Precision, Recall, F1-Score, and Matthews Correlation Coefficient.
- **Visual Confusion Matrix:** Annotated plot clearly distinguishing True Benign, False Malignant, False Benign, and True Malignant predictions.
- **Detailed Classification Report:** Per-class precision, recall, and F1 scores with support counts.
- **Side-by-Side Model Comparison:** Full comparison table evaluating all five models simultaneously on the uploaded dataset with performance highlights.
- **Dataset Inspector:** Expandable view to inspect raw test samples and underlying data distributions.
