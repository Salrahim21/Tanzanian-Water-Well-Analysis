
# 📘 Project: Binary Classification of Water Pump Functionality

This project applies machine learning techniques to classify the functionality status of water pumps in Tanzania. The dataset is converted into a binary classification problem: predicting whether a water pump is **functional (1)** or **non-functional/needs repair (0)**.

---

## 📁 Files

- `Notebook_Annotated.ipynb` — Jupyter Notebook containing the full workflow: preprocessing, resampling, training, tuning, and evaluation.

---

## 🔧 Problem Setup

- **Original Labels:**
  - `functional`
  - `functional needs repair`
  - `non functional`

- **Binary Mapping:**
  - `functional` → 1 (Positive Class)
  - `functional needs repair`, `non functional` → 0 (Negative Class)

---

## ✅ Methodology

1. **Preprocessing**
   - Convert sparse matrix features to DataFrame
   - Map target labels to binary

2. **Resampling**
   - Downsample the majority class (`functional`)
   - Upsample the minority class (`non-functional/needs repair`) using `resample` from `sklearn.utils`

3. **Modeling**
   - Train a `DecisionTreeClassifier` with class balancing
   - Tune hyperparameters using `GridSearchCV` with stratified 5-fold cross-validation
   - Train a `RandomForestClassifier` with grid search

4. **Evaluation**
   - Performance is evaluated on test data using:
     - **Classification Report:** Precision, Recall, F1-score
     - **Confusion Matrix**
   - Both macro and weighted averages reported

---

## 🔍 Example Results (Random Forest)

| Metric     | Class 0 (Functional) | Class 1 (Non-Functional) |
|------------|----------------------|---------------------------|
| Precision  | 0.80                 | 0.75                     |
| Recall     | 0.86                 | 0.66                     |
| F1-score   | 0.83                 | 0.70                     |
| Accuracy   | 78%                  |                           |

---

## 📈 Best Parameters Found (Random Forest)

```
{
  'max_depth': None,
  'min_samples_leaf': 1,
  'min_samples_split': 5,
  'n_estimators': 100
}
```

---

## 📌 Dependencies

- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib` / `seaborn` (optional for visualization)

---

## 📤 Future Work

- Improve recall for the minority class
- Explore ensemble models like `XGBoost`, `LightGBM`
- Incorporate ROC/AUC or precision-recall curves for better threshold selection
