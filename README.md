# 🚀 No-Code Advanced Machine Learning App (Streamlit)

An **end-to-end no-code Machine Learning platform** built with **Streamlit** that allows anyone to perform **complete ML workflows** — from dataset upload to advanced model training, evaluation, interpretation, and deployment.  

This project demonstrates how **advanced machine learning** can be made **accessible without writing code**, while still incorporating professional practices like **data cleaning, feature engineering, hyperparameter tuning, ensemble models, and explainability (XAI).**

---

## ✨ Features Overview

### 🔹 1. Dataset Upload & Data Info
- Users can upload their own **CSV/Excel dataset**.  
- The app automatically displays:
  - Shape of the dataset
  - Number of rows & columns
  - Data types of each column
  - Missing value counts

👉 *This ensures users understand their dataset before applying ML models.*

---

### 🔹 2. Data Restructuring
- Drop unnecessary columns
- Rename features
- Merge or split columns
- Define target variable

👉 *Gives flexibility to reshape raw data into a usable ML-ready dataset.*

---

### 🔹 3. Data Transformation
- **Encoding categorical features**:
  - One-Hot Encoding
  - Ordinal Encoding
  - Target Encoding
- **Scaling numerical features**:
  - StandardScaler
  - MinMaxScaler
  - RobustScaler
- **Imputation of missing values**:
  - Mean / Median
  - Mode
  - Custom values

👉 *Handles common preprocessing needs before training models.*

---

### 🔹 4. Data Issues Detection
- Identify:
  - Missing values
  - Duplicate records
  - Outliers
  - Class imbalance

👉 *Highlights data quality problems that affect model performance.*

---

### 🔹 5. Data Visualization
- **Histograms & Boxplots** → feature distributions  
- **Correlation Heatmaps** → feature relationships  
- **Pairplots** → multi-variable exploration  
- **Class balance plots** → check imbalance issues  

👉 *Enables intuitive understanding of data patterns.*

---

### 🔹 6. Quick Statistics
- Descriptive statistics (mean, median, std, min/max)
- Categorical summaries (counts, proportions)
- Target vs feature summaries

👉 *Quickly identifies trends and anomalies.*

---

### 🔹 7. Advanced Machine Learning

#### ⚙️ Setup
- Select target variable
- Train-test split (with stratification for classification)
- Define ML problem type: **Classification** or **Regression**

#### ⚖️ Imbalance Handling
- SMOTE (Synthetic Minority Oversampling)
- Random oversampling/undersampling
- Combination approaches

👉 *Prevents bias when datasets are imbalanced.*

#### 🧩 Feature Engineering
- Automatic feature selection (filter, wrapper, embedded methods)
- Correlation-based feature removal
- Custom feature transformations

#### 🤖 Model Training
- Supports multiple algorithms:
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - Gradient Boosting (XGBoost, LightGBM, CatBoost)
  - SVM
  - KNN
  - Linear/Polynomial Regression
- Trains models automatically with cross-validation

#### 🔍 Hyperparameter Tuning
- Uses **GridSearchCV** with **StratifiedKFold cross-validation**
- Finds best model parameters to maximize performance

#### 📉 Model Evaluation
- **Classification Metrics**: Accuracy, Precision, Recall, F1, ROC-AUC
- **Regression Metrics**: R², MAE, RMSE
- Visualizations:
  - Confusion Matrix
  - ROC Curves
  - Precision-Recall Curves
  - Residual Plots

#### 🔬 Interpretability (XAI)
- **Feature Importance** (tree-based models)
- **Permutation Importance**
- **Partial Dependence Plots**
- **Coefficients for linear models**
- **(Optional) SHAP explanations**

👉 *Gives users insight into how the model makes predictions.*

#### 🧑‍🤝‍🧑 Ensemble Models
- Bagging (Random Forest, Extra Trees)
- Boosting (XGBoost, LightGBM, AdaBoost)
- Stacking models
- Voting classifiers/regressors

👉 *Improves predictive accuracy using multiple models.*

#### 📦 Model Management
- Save trained models as `.pkl` files
- Reload saved models
- Download trained models for reuse

#### 🔮 Advanced Prediction
- Upload new datasets for prediction
- Generate predictions with probabilities
- Batch processing for multiple records

---

### 🔹 8. Deployment
- Fully integrated into a **Streamlit app**
- One-click launch using `streamlit run`
- Ready for deployment on **Streamlit Cloud, Heroku, or AWS**

---

## 🛠️ Tech Stack

- **Frontend/UI**: Streamlit  
- **Machine Learning**: scikit-learn, imbalanced-learn, XGBoost, LightGBM  
- **Data Processing**: pandas, numpy  
- **Visualization**: matplotlib, seaborn  
- **Explainability (XAI)**: SHAP, sklearn interpretability tools  

---

## 🚀 How to Run

1. Clone repository:
   ```bash
   git clone https://https://github.com/Samay9812/ML-Studio.git
   cd data_preview_app

