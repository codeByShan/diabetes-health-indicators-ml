# 🩺 Diabetes Health Indicators — ML Project

## 🚀 Live App

👉https://diabetes-health-indicators-ml.streamlit.app/

## 📌 Project Overview

A complete end-to-end Machine Learning project using the **Diabetes Health Indicators Dataset** containing **100,000 patient records** with clinical, lifestyle, and diagnostic data.

The project tackles three key ML tasks:
- 🔴 **Binary Classification:** Predict whether a patient has been diagnosed with diabetes (Yes/No)
- 🟡 **Multiclass Classification:** Predict the stage of diabetes (No Diabetes, Pre-Diabetes, Type 1, Type 2, Gestational)
- 🔵 **Regression:** Predict the continuous diabetes risk score

## 🛠️ Tech Stack
| Tool | Purpose |
|------|---------|
| Python | Core programming language |
| Pandas & NumPy | Data manipulation |
| Scikit-learn | ML models & preprocessing |
| Matplotlib & Seaborn | Data visualization |
| Streamlit | Interactive web app |
| Joblib | Model serialization |
| Git & GitHub | Version control |

## 📁 Project Structure

diabetes-health-indicators-ml/

├── data/

│   ├── raw/                        # Original CSV dataset

│   └── processed/                  # Cleaned & encoded data

├── notebooks/

│   └── main_analysis.ipynb         # Main Jupyter Notebook

├── models/

│   ├── binary_model.joblib         # Tuned Binary Classification model

│   ├── multiclass_model.joblib     # Tuned Multiclass Classification model

│   ├── regression_model.joblib     # Tuned Regression model

│   └── scaler.joblib               # StandardScaler

├── src/

│   ├── preprocessing.py            # Cleaning & encoding functions

│   └── evaluation.py               # Performance plotting functions

├── streamlit_app/

│   └── app.py                      # Main Streamlit interface

├── reports/                        # Saved visualizations & plots

├── requirements.txt                # Library dependencies

├── .gitignore                      # Files to exclude

└── README.md                       # Project guide


## 📊 Dataset
- **Source:** Diabetes Health Indicators Dataset
- **Records:** 100,000 patients
- **Features:** 31 columns including demographic, lifestyle, and clinical data
- **Target Variables:**
  - `diagnosed_diabetes` — Binary (0/1)
  - `diabetes_stage` — Multiclass (5 stages)
  - `diabetes_risk_score` — Continuous (2.7 to 67.2)

## 🔄 Project Workflow
1. ✅ **Repository Initialization** — GitHub repo & folder structure
2. ✅ **Data Loading & Inspection** — Shape, dtypes, null values
3. ✅ **Data Preprocessing** — Label encoding, standardization
4. ✅ **Exploratory Data Analysis** — Histograms, boxplots, heatmaps
5. ✅ **Data Splitting** — 80/20 stratified train-test split
6. ✅ **Baseline Model Training** — Logistic Regression, Decision Tree, KNN
7. ✅ **Hyperparameter Tuning** — GridSearchCV with cross-validation
8. ✅ **Streamlit Deployment** — Interactive web app on Streamlit Cloud

## 🤖 Models & Results

### 🔴 Binary Classification

| Model | Accuracy | F1-Score | ROC-AUC |
|-------|----------|----------|---------|
| Logistic Regression | Tuned ✅ | Tuned ✅ | Tuned ✅ |
| Decision Tree | Tuned ✅ | Tuned ✅ | Tuned ✅ |
| KNN | Baseline ✅ | Baseline ✅ | Baseline ✅ |

### 🟡 Multiclass Classification

| Model | Accuracy | Macro-F1 |
|-------|----------|----------|
| Logistic Regression | Baseline ✅ | Baseline ✅ |
| Decision Tree | Tuned ✅ | Tuned ✅ |
| KNN | Baseline ✅ | Baseline ✅ |

### 🔵 Regression

| Model | MAE | RMSE | R² |
|-------|-----|------|-----|
| Linear Regression | Baseline ✅ | Baseline ✅ | Baseline ✅ |
| Decision Tree Regressor | Tuned ✅ | Tuned ✅ | **0.97** ✅ |

## 🌐 Streamlit App Features

- 📋 **Sidebar** with all patient input fields
- 🔴 **Binary Tab** — Predicts Yes/No with confidence scores
- 🟡 **Multiclass Tab** — Predicts diabetes stage with probability chart
- 🔵 **Regression Tab** — Predicts risk score with Low/Moderate/High indicator

## ⚙️ How to Run Locally

```bash
# Clone the repo
git clone https://github.com/codeByShan/diabetes-health-indicators-ml.git
cd diabetes-health-indicators-ml

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run streamlit_app/app.py
```

## 👨‍💻 Author

**codeByShan**
Bootcamp / Certificate ML Project
