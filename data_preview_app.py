import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import os
import warnings
import pickle
from datetime import datetime
from category_encoders import TargetEncoder
import operator

# Try to import all sklearn components
try:
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder, OrdinalEncoder
    from sklearn.impute import SimpleImputer
    from sklearn.compose import ColumnTransformer
    from sklearn.model_selection import StratifiedKFold, KFold, GridSearchCV
    from sklearn.pipeline import Pipeline
    from sklearn.inspection import permutation_importance
    from sklearn.inspection import PartialDependenceDisplay
    from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
    from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    from sklearn.svm import SVC, SVR
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
    from sklearn.naive_bayes import GaussianNB
    from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                                roc_auc_score, mean_absolute_error, mean_squared_error, r2_score,
                                confusion_matrix, roc_curve)
    import joblib
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    st.warning("⚠️ scikit-learn not available. Machine learning features will be disabled.")

# Try to import imbalanced-learn
try:
    from imblearn.over_sampling import SMOTE, ADASYN
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.combine import SMOTETomek
    from imblearn.ensemble import BalancedBaggingClassifier
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False


# Check if XGBoost is installed
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

# Initialize session state for df, history, and redo
if 'df' not in st.session_state:
    st.session_state.df = None
if 'history' not in st.session_state:
    st.session_state.history = []
if 'redo_stack' not in st.session_state:
    st.session_state.redo_stack = []

# Central function to update df with history tracking
def update_df(new_df):
    """Update the dataframe while keeping history for undo/redo."""
    if st.session_state.df is not None:
        st.session_state.history.append(st.session_state.df.copy())  # save current state
    st.session_state.redo_stack = []  # clear redo stack on new action
    st.session_state.df = new_df



warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

# ================================
# Page Config
# ================================
st.set_page_config(page_title="Data Prep + EDA + AI Assistant", page_icon="📊", layout="wide")
st.title("📊 ML-Studio")
st.markdown("Upload your dataset, check data quality, clean data, explore visually, and get AI insights!")

# ================================
# Helper Functions
# ================================
@st.cache_data
def load_data(file):
    """Load data with error handling and caching"""
    try:
        if file.name.endswith('.csv'):
            # Try different encodings and separators
            try:
                df = pd.read_csv(file, encoding='utf-8')
            except UnicodeDecodeError:
                file.seek(0)
                df = pd.read_csv(file, encoding='latin-1')
            except pd.errors.ParserError:
                file.seek(0)
                df = pd.read_csv(file, sep=';', encoding='utf-8')
        elif file.name.endswith(('.xlsx', 'xls')):
            df = pd.read_excel(file, engine='openpyxl' if file.name.endswith('.xlsx') else 'xlrd')
        elif file.name.endswith('.json'):
            df = pd.read_json(file)
        elif file.name.endswith('.parquet'):
            df = pd.read_parquet(file)
        else:
            st.error("Unsupported file format")
            return None
        
        # Basic data validation
        if df.empty:
            st.error("The uploaded file is empty")
            return None
        if len(df.columns) == 0:
            st.error("No columns found in the dataset")
            return None
            
        return df
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

def get_column_types(df):
    """Categorize columns by data type"""
    numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
    categorical_cols = list(df.select_dtypes(include=['object', 'category']).columns)
    datetime_cols = list(df.select_dtypes(include=['datetime64']).columns)
    boolean_cols = list(df.select_dtypes(include=['bool']).columns)
    
    return {
        'numeric': numeric_cols,
        'categorical': categorical_cols,
        'datetime': datetime_cols,
        'boolean': boolean_cols
    }

def detect_outliers(df, column):
    """Detect outliers using IQR method"""
    if column not in df.columns or not pd.api.types.is_numeric_dtype(df[column]):
        return None
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return len(outliers), lower_bound, upper_bound

def safe_plot(plot_func, *args, **kwargs):
    """Wrapper for safe plotting with error handling"""
    try:
        return plot_func(*args, **kwargs)
    except Exception as e:
        st.error(f"Error creating plot: {str(e)}")
        return None

def analyze_target_variable(df, target_col):
    """Analyze target variable to determine problem type"""
    target_data = df[target_col]
    unique_count = target_data.nunique()
    
    # Determine if it's classification or regression
    if target_data.dtype == 'object' or target_data.dtype.name == 'category':
        problem_type = 'Classification'
    elif unique_count <= 10 and target_data.dtype in ['int64', 'int32', 'bool']:
        problem_type = 'Classification'
    else:
        problem_type = 'Regression'
    
    return {
        'type': problem_type,
        'unique_count': unique_count,
        'dtype': str(target_data.dtype)
    }

def auto_feature_selection(df, target_col, corr_threshold=0.8, max_features=20):
    """
    Automatically select features based on correlation with target and between features.
    
    Args:
        df (pd.DataFrame): Input dataset
        target_col (str): Target column name
        corr_threshold (float): Maximum allowed correlation between features
        max_features (int): Maximum number of features to return
    
    Returns:
        selected_features (list): List of selected feature column names
    """
    # Numeric columns only for correlation analysis
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if target_col in numeric_cols:
        # Compute correlation matrix
        corr_matrix = df[numeric_cols].corr()
        target_corr = abs(corr_matrix[target_col]).sort_values(ascending=False)
        
        # Select features correlated with target but not too correlated with each other
        selected_features = []
        for feature in target_corr.index:
            if feature == target_col:
                continue
            
            if not selected_features:
                selected_features.append(feature)
            else:
                max_corr = max([abs(corr_matrix[feature][sf]) for sf in selected_features])
                if max_corr < corr_threshold:
                    selected_features.append(feature)
    else:
        # If target is not numeric, just return all numeric features
        selected_features = [col for col in numeric_cols if col != target_col]

    # Add categorical features
    categorical_cols = [col for col in df.columns if col not in numeric_cols and col != target_col]
    selected_features.extend(categorical_cols)
    
    # Limit total number of features
    return selected_features[:max_features]

def train_multiple_models(X_train, y_train, X_test, y_test, selected_models, 
                         available_models, problem_type, scaling_method, 
                         encoding_method, numeric_imputation, categorical_imputation, cv_folds):
    """Train multiple models and return results"""
    if not SKLEARN_AVAILABLE:
        st.error("scikit-learn is required for model training")
        return {}
    
    results = {}
    
    # Get column types
    numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Create preprocessing steps
    preprocessor_steps = []
    
    if numeric_features:
        numeric_steps = [('imputer', SimpleImputer(strategy=numeric_imputation))]
        if scaling_method != 'None':
            if scaling_method == 'StandardScaler':
                numeric_steps.append(('scaler', StandardScaler()))
            elif scaling_method == 'MinMaxScaler':
                numeric_steps.append(('scaler', MinMaxScaler()))
        
        numeric_transformer = Pipeline(numeric_steps)
        preprocessor_steps.append(('num', numeric_transformer, numeric_features))
    
    if categorical_features:
        categorical_steps = [('imputer', SimpleImputer(strategy=categorical_imputation, fill_value='Unknown'))]
        if encoding_method == 'OneHotEncoder':
            categorical_steps.append(('encoder', OneHotEncoder(drop='first', sparse=False, handle_unknown='ignore')))
        elif encoding_method == 'LabelEncoder':
            # Note: LabelEncoder in pipeline is tricky, using OrdinalEncoder instead
            categorical_steps.append(('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)))
        
        categorical_transformer = Pipeline(categorical_steps)
        preprocessor_steps.append(('cat', categorical_transformer, categorical_features))
    
    # Create preprocessor
    if preprocessor_steps:
        preprocessor = ColumnTransformer(preprocessor_steps, remainder='passthrough')
    else:
        preprocessor = 'passthrough'
    
    # Train each selected model
    for model_name in selected_models:
        model_code = available_models[model_name]
        
        try:
            # Get the model
            if problem_type == 'Classification':
                if model_code == 'lr':
                    model = LogisticRegression(random_state=42, max_iter=1000)
                elif model_code == 'rf':
                    model = RandomForestClassifier(random_state=42, n_estimators=100)
                elif model_code == 'dt':
                    model = DecisionTreeClassifier(random_state=42)
                elif model_code == 'gb':
                    model = GradientBoostingClassifier(random_state=42)
                elif model_code == 'svm':
                    model = SVC(random_state=42, probability=True)
                elif model_code == 'knn':
                    model = KNeighborsClassifier()
                elif model_code == 'nb':
                    model = GaussianNB()
                elif model_code == 'xgb' and XGBOOST_AVAILABLE:
                    model = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
                elif model_code == 'lgb' and LIGHTGBM_AVAILABLE:
                    model = lgb.LGBMClassifier(random_state=42, verbosity=-1)
                else:
                    continue
            else:  # Regression
                if model_code == 'lr':
                    model = LinearRegression()
                elif model_code == 'rf':
                    model = RandomForestRegressor(random_state=42, n_estimators=100)
                elif model_code == 'dt':
                    model = DecisionTreeRegressor(random_state=42)
                elif model_code == 'gb':
                    model = GradientBoostingRegressor(random_state=42)
                elif model_code == 'ridge':
                    model = Ridge(random_state=42)
                elif model_code == 'lasso':
                    model = Lasso(random_state=42)
                elif model_code == 'svm':
                    model = SVR()
                elif model_code == 'knn':
                    model = KNeighborsRegressor()
                elif model_code == 'xgb' and XGBOOST_AVAILABLE:
                    model = xgb.XGBRegressor(random_state=42)
                elif model_code == 'lgb' and LIGHTGBM_AVAILABLE:
                    model = lgb.LGBMRegressor(random_state=42, verbosity=-1)
                else:
                    continue
            
            # Create pipeline
            if problem_type == 'Classification':
                pipeline = Pipeline([
                    ('preprocessor', preprocessor),
                    ('classifier', model)
                ])
            else:
                pipeline = Pipeline([
                    ('preprocessor', preprocessor),
                    ('regressor', model)
                ])
            
            # Fit the model
            pipeline.fit(X_train, y_train)
            
            # Make predictions
            y_pred = pipeline.predict(X_test)
            
            # Calculate metrics
            if problem_type == 'Classification':
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                
                try:
                    # Check if model can predict probabilities
                    if hasattr(pipeline, 'predict_proba'):
                        y_proba = pipeline.predict_proba(X_test)
                        # Check if binary classification
                        if len(np.unique(y_test)) == 2 and y_proba.shape[1] == 2:
                            auc = roc_auc_score(y_test, y_proba[:, 1])
                        else:  # Multi-class
                            auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted')
                    else:
                        auc = 0.0
                except Exception as e:
                    auc = 0.0
                
                # Cross-validation
                cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv_folds, scoring='f1_weighted')
                
                results[model_name] = {
                    'model': pipeline,
                    'predictions': y_pred,
                    'test_accuracy': accuracy,
                    'test_precision': precision,
                    'test_recall': recall,
                    'test_f1': f1,
                    'test_auc': auc,
                    'cv_scores': cv_scores
                }
                
            else:  # Regression
                mae = mean_absolute_error(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test, y_pred)
                
                # Cross-validation
                cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv_folds, scoring='r2')
                
                results[model_name] = {
                    'model': pipeline,
                    'predictions': y_pred,
                    'test_mae': mae,
                    'test_mse': mse,
                    'test_rmse': rmse,
                    'test_r2': r2,
                    'cv_scores': cv_scores
                }
            
        except Exception as e:
            st.warning(f"Failed to train {model_name}: {str(e)}")
            continue
    
    return results

def create_comparison_table(results, problem_type):
    """Create model comparison table"""
    comparison_data = []
    
    for model_name, model_data in results.items():
        row = {'Model': model_name}
        
        if problem_type == 'Classification':
            row.update({
                'Accuracy': f"{model_data['test_accuracy']:.3f}",
                'Precision': f"{model_data['test_precision']:.3f}",
                'Recall': f"{model_data['test_recall']:.3f}",
                'F1 Score': f"{model_data['test_f1']:.3f}",
                'AUC': f"{model_data['test_auc']:.3f}",
                'CV Mean': f"{model_data['cv_scores'].mean():.3f}",
                'CV Std': f"{model_data['cv_scores'].std():.3f}"
            })
        else:
            row.update({
                'MAE': f"{model_data['test_mae']:.3f}",
                'MSE': f"{model_data['test_mse']:.3f}",
                'RMSE': f"{model_data['test_rmse']:.3f}",
                'R² Score': f"{model_data['test_r2']:.3f}",
                'CV Mean': f"{model_data['cv_scores'].mean():.3f}",
                'CV Std': f"{model_data['cv_scores'].std():.3f}"
            })
        
        comparison_data.append(row)
    
    return pd.DataFrame(comparison_data)

def get_feature_names_from_pipeline(pipeline, original_features):
    """Extract feature names from a sklearn pipeline"""
    try:
        # Try to get feature names from the preprocessor
        if hasattr(pipeline.named_steps['preprocessor'], 'get_feature_names_out'):
            feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()
        else:
            # Fallback to original feature names
            feature_names = original_features
        
        # Convert to list and clean up names
        feature_names = [str(name).replace('num__', '').replace('cat__', '') for name in feature_names]
        return feature_names
    except:
        return original_features

import shap
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.base import BaseEstimator

# Define tree-based model types for detection
TREE_MODELS = (RandomForestClassifier, GradientBoostingClassifier)

# Define known tree-based models
TREE_MODELS = (RandomForestClassifier, GradientBoostingClassifier)

# Define known tree-based models
TREE_MODELS = (RandomForestClassifier, GradientBoostingClassifier)

def extract_raw_model(model):
    """Safely unwrap model from pipeline or nested structure."""
    if isinstance(model, Pipeline):
        for step_name, step_obj in model.named_steps.items():
            if hasattr(step_obj, "predict"):
                return step_obj
    return getattr(model, 'estimator', model)

import shap
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

# Define tree-based models for TreeExplainer compatibility
TREE_MODELS = (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    DecisionTreeClassifier, DecisionTreeRegressor
)

def safe_extract_model_from_pipeline(pipeline_model):
    """
    Safely extract the actual model from a sklearn pipeline
    """
    if not isinstance(pipeline_model, Pipeline):
        return pipeline_model
    
    # Look for the final estimator (classifier/regressor)
    for step_name, step_obj in reversed(pipeline_model.steps):
        if hasattr(step_obj, 'predict'):
            return step_obj
    
    return pipeline_model

def prepare_data_for_shap(X_data, preprocessor=None):
    """
    Prepare data for SHAP analysis with comprehensive preprocessing
    """
    try:
        X_processed = X_data.copy()
        
        # 1. Handle datetime columns
        datetime_cols = X_processed.select_dtypes(include=['datetime64']).columns
        if len(datetime_cols) > 0:
            st.info(f"Removing {len(datetime_cols)} datetime columns for SHAP analysis")
            X_processed = X_processed.drop(columns=datetime_cols)
        
        # 2. Handle boolean columns - convert to int
        bool_cols = X_processed.select_dtypes(include=['bool']).columns
        for col in bool_cols:
            X_processed[col] = X_processed[col].astype(int)
        
        # 3. Handle categorical columns
        categorical_cols = X_processed.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            if X_processed[col].dtype.name == 'category':
                # Use existing categorical codes
                X_processed[col] = X_processed[col].cat.codes
            else:
                # Convert object to categorical codes
                X_processed[col] = pd.Categorical(X_processed[col]).codes
        
        # 4. Handle missing values
        X_processed = X_processed.fillna(X_processed.mean(numeric_only=True))
        
        # Fill any remaining NaNs with 0
        X_processed = X_processed.fillna(0)
        
        # 5. Ensure all data is numeric
        for col in X_processed.columns:
            if not pd.api.types.is_numeric_dtype(X_processed[col]):
                try:
                    X_processed[col] = pd.to_numeric(X_processed[col], errors='coerce')
                    X_processed[col] = X_processed[col].fillna(0)
                except:
                    # If conversion fails, create dummy encoding
                    X_processed[col] = pd.Categorical(X_processed[col]).codes
        
        # 6. Convert to float to ensure consistency
        X_processed = X_processed.astype(float)
        
        # 7. Handle infinite values
        X_processed = X_processed.replace([np.inf, -np.inf], 0)
        
        return X_processed
    
    except Exception as e:
        st.error(f"Data preparation failed: {str(e)}")
        return None

def create_pipeline_predict_function(pipeline, X_sample_original):
    """
    Create a prediction function that handles the full pipeline
    """
    def predict_func(X_array):
        try:
            # Convert numpy array back to DataFrame with original column names
            X_df = pd.DataFrame(X_array, columns=X_sample_original.columns)
            return pipeline.predict(X_df)
        except Exception as e:
            # Fallback: try with just the array
            return pipeline.predict(X_array)
    
    return predict_func

def generate_robust_shap_analysis(model, X_sample, model_name, problem_type):
    """
    Robust SHAP analysis with multiple fallback strategies
    """
    try:
        st.write(f"**🎯 SHAP Analysis for {model_name}**")
        
        # Step 1: Prepare sample data
        sample_size = min(50, len(X_sample))  # Keep it small for performance
        X_shap_original = X_sample.sample(n=sample_size, random_state=42) if len(X_sample) > sample_size else X_sample.copy()
        
        # Step 2: Prepare data for SHAP
        X_shap_processed = prepare_data_for_shap(X_shap_original)
        if X_shap_processed is None:
            return
        
        # Step 3: Extract the actual model
        actual_model = safe_extract_model_from_pipeline(model)
        
        # Step 4: Choose SHAP explainer strategy
        explainer = None
        shap_values = None
        explanation_method = "Unknown"
        
        # Strategy 1: TreeExplainer for tree-based models
        if isinstance(actual_model, TREE_MODELS):
            try:
                st.info("Using TreeExplainer for tree-based model")
                explainer = shap.TreeExplainer(
                    actual_model, 
                    feature_perturbation='interventional',
                    check_additivity=False
                )
                shap_values = explainer(X_shap_processed)
                explanation_method = "TreeExplainer"
                
            except Exception as tree_error:
                st.warning(f"TreeExplainer failed: {tree_error}")
                explainer = None
        
        # Strategy 2: LinearExplainer for linear models
        if explainer is None and hasattr(actual_model, 'coef_'):
            try:
                st.info("Using LinearExplainer for linear model")
                explainer = shap.LinearExplainer(actual_model, X_shap_processed)
                shap_values = explainer.shap_values(X_shap_processed)
                explanation_method = "LinearExplainer"
                
            except Exception as linear_error:
                st.warning(f"LinearExplainer failed: {linear_error}")
                explainer = None
        
        # Strategy 3: KernelExplainer as fallback (works with pipelines)
        if explainer is None:
            try:
                st.info("Using KernelExplainer (may take longer)")
                # Use smaller background for KernelExplainer
                background_size = min(10, len(X_shap_processed))
                background = X_shap_processed.sample(n=background_size, random_state=42)
                
                # Create prediction function
                if isinstance(model, Pipeline):
                    predict_func = create_pipeline_predict_function(model, X_shap_original)
                    explainer = shap.KernelExplainer(predict_func, background.values)
                    shap_values = explainer.shap_values(X_shap_processed.values)
                else:
                    explainer = shap.KernelExplainer(model.predict, background)
                    shap_values = explainer.shap_values(X_shap_processed)
                
                explanation_method = "KernelExplainer"
                
            except Exception as kernel_error:
                st.error(f"All SHAP explainers failed. KernelExplainer error: {kernel_error}")
                return
        
        # Strategy 4: Permutation-based explainer as final fallback
        if explainer is None or shap_values is None:
            try:
                st.info("Using Permutation-based explainer as final fallback")
                explainer = shap.PermutationExplainer(model.predict, X_shap_processed)
                shap_values = explainer(X_shap_processed)
                explanation_method = "PermutationExplainer"
                
            except Exception as perm_error:
                st.error(f"Final fallback failed: {perm_error}")
                return
        
        # Step 5: Generate visualizations
        if shap_values is not None:
            st.success(f"✅ SHAP analysis successful using {explanation_method}")
            
            # Summary plot
            try:
                st.write("**📊 SHAP Summary Plot**")
                fig_summary, ax_summary = plt.subplots(figsize=(10, 8))
                
                # Handle different shap_values formats
                if isinstance(shap_values, list):
                    # Multi-class classification
                    plot_values = shap_values[0] if len(shap_values) > 0 else shap_values
                    plot_data = X_shap_processed
                elif hasattr(shap_values, 'values'):
                    # New SHAP format (Explanation object)
                    plot_values = shap_values.values
                    plot_data = X_shap_processed
                else:
                    # Direct numpy array
                    plot_values = shap_values
                    plot_data = X_shap_processed
                
                shap.summary_plot(
                    plot_values, 
                    plot_data,
                    feature_names=X_shap_processed.columns,
                    show=False,
                    max_display=15
                )
                
                st.pyplot(fig_summary)
                plt.close(fig_summary)
                
            except Exception as summary_error:
                st.warning(f"Summary plot failed: {summary_error}")
            
            # Waterfall plot for first prediction
            try:
                st.write("**🌊 SHAP Waterfall Plot (First Sample)**")
                fig_waterfall, ax_waterfall = plt.subplots(figsize=(10, 6))
                
                if hasattr(shap_values, 'values') and hasattr(shap_values, 'base_values'):
                    # New SHAP format
                    shap.waterfall_plot(shap_values[0], show=False)
                elif isinstance(shap_values, list):
                    # Multi-class - use first class
                    if hasattr(explainer, 'expected_value'):
                        expected_value = explainer.expected_value[0] if isinstance(explainer.expected_value, list) else explainer.expected_value
                    else:
                        expected_value = 0
                    
                    shap.waterfall_plot(
                        shap.Explanation(
                            values=shap_values[0][0],
                            base_values=expected_value,
                            data=X_shap_processed.iloc[0].values,
                            feature_names=X_shap_processed.columns
                        ),
                        show=False
                    )
                else:
                    # Direct values
                    if hasattr(explainer, 'expected_value'):
                        expected_value = explainer.expected_value
                    else:
                        expected_value = 0
                    
                    shap.waterfall_plot(
                        shap.Explanation(
                            values=shap_values[0],
                            base_values=expected_value,
                            data=X_shap_processed.iloc[0].values,
                            feature_names=X_shap_processed.columns
                        ),
                        show=False
                    )
                
                st.pyplot(fig_waterfall)
                plt.close(fig_waterfall)
                
            except Exception as waterfall_error:
                st.warning(f"Waterfall plot failed: {waterfall_error}")
            
            # Feature importance plot
            try:
                st.write("**📈 SHAP Feature Importance**")
                fig_importance, ax_importance = plt.subplots(figsize=(10, 6))
                
                # Calculate mean absolute SHAP values
                if isinstance(shap_values, list):
                    importance_values = np.abs(shap_values[0]).mean(axis=0)
                elif hasattr(shap_values, 'values'):
                    importance_values = np.abs(shap_values.values).mean(axis=0)
                else:
                    importance_values = np.abs(shap_values).mean(axis=0)
                
                # Create importance DataFrame
                importance_df = pd.DataFrame({
                    'Feature': X_shap_processed.columns,
                    'Importance': importance_values
                }).sort_values('Importance', ascending=True)
                
                # Plot top 15 features
                top_features = importance_df.tail(15)
                ax_importance.barh(range(len(top_features)), top_features['Importance'])
                ax_importance.set_yticks(range(len(top_features)))
                ax_importance.set_yticklabels(top_features['Feature'])
                ax_importance.set_xlabel('Mean |SHAP Value|')
                ax_importance.set_title('Feature Importance (Mean |SHAP Value|)')
                ax_importance.grid(True, alpha=0.3)
                
                st.pyplot(fig_importance)
                plt.close(fig_importance)
                
            except Exception as importance_error:
                st.warning(f"Feature importance plot failed: {importance_error}")
        
    except Exception as e:
        st.error(f"SHAP analysis completely failed: {str(e)}")
        st.info("Consider using the alternative feature importance methods instead")

# Updated function to replace in your code
def generate_shap_analysis(model, X_sample, model_name, problem_type):
    """
    Main SHAP analysis function - updated and robust version
    """
    generate_robust_shap_analysis(model, X_sample, model_name, problem_type)

def create_model_report(results, problem_type):
    """Create comprehensive model results report"""
    report_data = []
    
    for model_name, model_data in results.items():
        row = {'Model': model_name}
        
        if problem_type == 'Classification':
            row.update({
                'Accuracy': model_data['test_accuracy'],
                'Precision': model_data['test_precision'],
                'Recall': model_data['test_recall'],
                'F1_Score': model_data['test_f1'],
                'AUC': model_data['test_auc'],
                'CV_Mean': model_data['cv_scores'].mean(),
                'CV_Std': model_data['cv_scores'].std()
            })
        else:
            row.update({
                'MAE': model_data['test_mae'],
                'MSE': model_data['test_mse'],
                'RMSE': model_data['test_rmse'],
                'R2_Score': model_data['test_r2'],
                'CV_Mean': model_data['cv_scores'].mean(),
                'CV_Std': model_data['cv_scores'].std()
            })
        
        report_data.append(row)
    
    report_df = pd.DataFrame(report_data)
    
    # Save to buffer
    buffer = BytesIO()
    report_df.to_csv(buffer, index=False)
    buffer.seek(0)
    
    return buffer

def check_class_imbalance(y):
    """Check for class imbalance in target variable"""
    if len(y.value_counts()) < 2:
        return False, None
    
    class_counts = y.value_counts()
    majority_count = class_counts.max()
    minority_count = class_counts.min()
    imbalance_ratio = majority_count / minority_count
    
    # Consider imbalanced if ratio > 2
    is_imbalanced = imbalance_ratio > 2
    
    return is_imbalanced, {
        'ratio': imbalance_ratio,
        'majority_class': class_counts.idxmax(),
        'minority_class': class_counts.idxmin(),
        'majority_count': majority_count,
        'minority_count': minority_count
    }

def handle_imbalanced_data(X, y, method='smote'):
    """Handle imbalanced data using various techniques"""
    if not IMBLEARN_AVAILABLE:
        st.error("imbalanced-learn package required for handling imbalanced data")
        return X, y
    
    try:
        if method == 'smote':
            sampler = SMOTE(random_state=42)
        elif method == 'adasyn':
            sampler = ADASYN(random_state=42)
        elif method == 'undersample':
            sampler = RandomUnderSampler(random_state=42)
        elif method == 'smote_tomek':
            sampler = SMOTETomek(random_state=42)
        else:
            return X, y
        
        X_resampled, y_resampled = sampler.fit_resample(X, y)
        return X_resampled, y_resampled
    except Exception as e:
        st.error(f"Error in resampling: {str(e)}")
        return X, y

def create_interaction_features(df, feature_pairs=None, max_interactions=10):
    """Create interaction features between numeric columns"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) < 2:
        return df
    
    interaction_df = df.copy()
    interactions_created = 0
    
    if feature_pairs:
        # Use specified feature pairs
        for col1, col2 in feature_pairs:
            if col1 in numeric_cols and col2 in numeric_cols:
                interaction_df[f"{col1}_{col2}_interaction"] = df[col1] * df[col2]
                interactions_created += 1
    else:
        # Create interactions between top correlated features
        for i in range(len(numeric_cols)):
            for j in range(i+1, len(numeric_cols)):
                if interactions_created >= max_interactions:
                    break
                col1, col2 = numeric_cols[i], numeric_cols[j]
                interaction_df[f"{col1}_{col2}_interaction"] = df[col1] * df[col2]
                interactions_created += 1
    
    return interaction_df

def advanced_feature_selection(X, y, method='correlation', n_features=20):
    """Advanced feature selection methods"""
    if method == 'correlation' and len(X.select_dtypes(include=[np.number]).columns) > 0:
        # Correlation-based selection
        numeric_X = X.select_dtypes(include=[np.number])
        if y.dtype in ['object', 'category']:
            # For categorical target, use chi-square or mutual information
            from sklearn.feature_selection import SelectKBest, chi2
            selector = SelectKBest(chi2, k=min(n_features, len(numeric_X.columns)))
            X_selected = selector.fit_transform(numeric_X, y)
            selected_features = numeric_X.columns[selector.get_support()]
            return X[selected_features]
        else:
            # For numeric target, use correlation
            corr_with_target = numeric_X.corrwith(y).abs().sort_values(ascending=False)
            selected_features = corr_with_target.head(n_features).index
            return X[selected_features]
    
    return X

def perform_hyperparameter_tuning(model, X, y, param_grid, cv=5, method='grid'):
    """Perform hyperparameter tuning"""
    try:
        if method == 'grid':
            search = GridSearchCV(model, param_grid, cv=cv, scoring='f1_weighted', n_jobs=-1)
        elif method == 'random':
            search = RandomizedSearchCV(model, param_grid, cv=cv, scoring='f1_weighted', 
                                      n_iter=20, n_jobs=-1, random_state=42)
        
        search.fit(X, y)
        return search.best_estimator_, search.best_params_, search.best_score_
    except Exception as e:
        st.warning(f"Hyperparameter tuning failed: {str(e)}")
        return model, {}, 0.0

def create_ensemble_model(models, X, y, method='voting'):
    """Create ensemble models"""
    try:
        if method == 'voting':
            estimators = [(name, model) for name, model in models.items()]
            ensemble = VotingClassifier(estimators=estimators, voting='soft')
        
        ensemble.fit(X, y)
        return ensemble
    except Exception as e:
        st.warning(f"Ensemble creation failed: {str(e)}")
        return None

def calculate_permutation_importance(model, X, y, n_repeats=10):
    """Calculate permutation importance as SHAP alternative"""
    try:
        perm_importance = permutation_importance(model, X, y, n_repeats=n_repeats, random_state=42)
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': perm_importance.importances_mean,
            'std': perm_importance.importances_std
        }).sort_values('importance', ascending=False)
        return importance_df
    except Exception as e:
        st.error(f"Permutation importance calculation failed: {str(e)}")
        return None

def create_partial_dependence_plots(model, X, features, n_features=4):
    """Create partial dependence plots"""
    try:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.ravel()
        
        selected_features = features[:n_features]
        
        for i, feature in enumerate(selected_features):
            if i >= 4:
                break
            
            display = PartialDependenceDisplay.from_estimator(
                model, X, [feature], ax=axes[i], kind='average'
            )
            axes[i].set_title(f'Partial Dependence: {feature}')
        
        # Hide empty subplots
        for i in range(len(selected_features), 4):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        return fig
    except Exception as e:
        st.error(f"Partial dependence plots failed: {str(e)}")
        return None

def train_advanced_models(X_train, y_train, X_test, y_test, selected_models, 
                         problem_type, use_class_weights=False, tune_hyperparams=False):
    """Train models with advanced features"""
    results = {}
    
    # Define parameter grids for hyperparameter tuning
    param_grids = {
        'RandomForest': {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10]
        },
        'XGBoost': {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.1, 0.2]
        },
        'LightGBM': {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.1, 0.2]
        }
    }
    
    for model_name in selected_models:
        try:
            # Initialize model
            if problem_type == 'Classification':
                if model_name == 'LogisticRegression':
                    model = LogisticRegression(random_state=42, max_iter=1000, 
                                             class_weight='balanced' if use_class_weights else None)
                elif model_name == 'RandomForest':
                    model = RandomForestClassifier(random_state=42, n_estimators=100,
                                                 class_weight='balanced' if use_class_weights else None)
                elif model_name == 'XGBoost' and XGBOOST_AVAILABLE:
                    scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1]) if use_class_weights else 1
                    model = xgb.XGBClassifier(random_state=42, eval_metric='logloss',
                                            scale_pos_weight=scale_pos_weight)
                elif model_name == 'LightGBM' and LIGHTGBM_AVAILABLE:
                    model = lgb.LGBMClassifier(random_state=42, verbosity=-1,
                                             class_weight='balanced' if use_class_weights else None)
                elif model_name == 'CatBoost' and CATBOOST_AVAILABLE:
                    model = cb.CatBoostClassifier(random_state=42, verbose=False,
                                                class_weights='Balanced' if use_class_weights else None)
                elif model_name == 'BalancedBagging' and IMBLEARN_AVAILABLE:
                    model = BalancedBaggingClassifier(random_state=42)
                else:
                    continue
            else:
                # Regression models
                if model_name == 'LinearRegression':
                    model = LinearRegression()
                elif model_name == 'RandomForest':
                    model = RandomForestRegressor(random_state=42, n_estimators=100)
                elif model_name == 'XGBoost' and XGBOOST_AVAILABLE:
                    model = xgb.XGBRegressor(random_state=42)
                elif model_name == 'LightGBM' and LIGHTGBM_AVAILABLE:
                    model = lgb.LGBMRegressor(random_state=42, verbosity=-1)
                elif model_name == 'CatBoost' and CATBOOST_AVAILABLE:
                    model = cb.CatBoostRegressor(random_state=42, verbose=False)
                else:
                    continue
            
            # Hyperparameter tuning
            if tune_hyperparams and model_name in param_grids:
                st.info(f"Tuning hyperparameters for {model_name}...")
                model, best_params, best_score = perform_hyperparameter_tuning(
                    model, X_train, y_train, param_grids[model_name]
                )
            else:
                best_params = {}
                best_score = 0.0
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            if problem_type == 'Classification':
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                
                try:
                    if hasattr(model, 'predict_proba'):
                        y_proba = model.predict_proba(X_test)
                        if len(np.unique(y_test)) == 2 and y_proba.shape[1] == 2:
                            auc = roc_auc_score(y_test, y_proba[:, 1])
                            pr_auc = average_precision_score(y_test, y_proba[:, 1])
                        else:
                            auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted')
                            pr_auc = 0.0  # PR-AUC for multiclass is complex
                    else:
                        auc = 0.0
                        pr_auc = 0.0
                except Exception:
                    auc = 0.0
                    pr_auc = 0.0
                
                # Cross-validation with stratification
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1_weighted')
                
                results[model_name] = {
                    'model': model,
                    'predictions': y_pred,
                    'test_accuracy': accuracy,
                    'test_precision': precision,
                    'test_recall': recall,
                    'test_f1': f1,
                    'test_auc': auc,
                    'test_pr_auc': pr_auc,
                    'cv_scores': cv_scores,
                    'best_params': best_params,
                    'tuning_score': best_score
                }
                
            else:  # Regression
                mae = mean_absolute_error(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test, y_pred)
                
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
                
                results[model_name] = {
                    'model': model,
                    'predictions': y_pred,
                    'test_mae': mae,
                    'test_mse': mse,
                    'test_rmse': rmse,
                    'test_r2': r2,
                    'cv_scores': cv_scores,
                    'best_params': best_params,
                    'tuning_score': best_score
                }
            
        except Exception as e:
            st.warning(f"Failed to train {model_name}: {str(e)}")
            continue
    
    return results

def create_advanced_comparison_table(results, problem_type):
    """Create enhanced model comparison table"""
    comparison_data = []
    
    for model_name, model_data in results.items():
        row = {'Model': model_name}
        
        if problem_type == 'Classification':
            row.update({
                'Accuracy': f"{model_data['test_accuracy']:.3f}",
                'Precision': f"{model_data['test_precision']:.3f}",
                'Recall': f"{model_data['test_recall']:.3f}",
                'F1 Score': f"{model_data['test_f1']:.3f}",
                'ROC-AUC': f"{model_data['test_auc']:.3f}",
                'PR-AUC': f"{model_data['test_pr_auc']:.3f}",
                'CV Mean': f"{model_data['cv_scores'].mean():.3f}",
                'CV Std': f"{model_data['cv_scores'].std():.3f}",
                'Tuning Score': f"{model_data.get('tuning_score', 0):.3f}"
            })
        else:
            row.update({
                'MAE': f"{model_data['test_mae']:.3f}",
                'MSE': f"{model_data['test_mse']:.3f}",
                'RMSE': f"{model_data['test_rmse']:.3f}",
                'R² Score': f"{model_data['test_r2']:.3f}",
                'CV Mean': f"{model_data['cv_scores'].mean():.3f}",
                'CV Std': f"{model_data['cv_scores'].std():.3f}",
                'Tuning Score': f"{model_data.get('tuning_score', 0):.3f}"
            })
        
        comparison_data.append(row)
    
    return pd.DataFrame(comparison_data)

def generate_classification_report_plot(y_true, y_pred, classes):
    """Generate classification report as a heatmap"""
    report = classification_report(y_true, y_pred, target_names=classes, output_dict=True)
    
    # Extract metrics for each class
    metrics_data = []
    for class_name in classes:
        if class_name in report:
            metrics_data.append([
                report[class_name]['precision'],
                report[class_name]['recall'],
                report[class_name]['f1-score']
            ])
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(metrics_data, annot=True, fmt='.3f', cmap='YlOrRd',
                xticklabels=['Precision', 'Recall', 'F1-Score'],
                yticklabels=classes, ax=ax)
    ax.set_title('Classification Report Heatmap')
    return fig

def generate_ai_response(query, context, df):
    """Generate AI-like response based on query and dataset context"""
    
    query_lower = query.lower()
    
    # Pattern matching for common queries
    if "pattern" in query_lower or "insight" in query_lower:
        response = f"""Based on your dataset analysis, here are the key patterns I found:

**Dataset Overview:**
- Shape: {context['shape'][0]:,} rows × {context['shape'][1]} columns
- Data types: {len(context['numeric_columns'])} numeric, {len(context['categorical_columns'])} categorical

**Key Patterns:**
"""
        if context['numeric_columns']:
            response += f"- Numeric features: {', '.join(context['numeric_columns'][:3])}{'...' if len(context['numeric_columns']) > 3 else ''}\n"
        
        if context['categorical_columns']:
            response += f"- Categorical features: {', '.join(context['categorical_columns'][:3])}{'...' if len(context['categorical_columns']) > 3 else ''}\n"
        
        # Missing value patterns
        missing_cols = [col for col, count in context['missing_values'].items() if count > 0]
        if missing_cols:
            response += f"- Missing data in: {', '.join(missing_cols[:3])}{'...' if len(missing_cols) > 3 else ''}\n"
        
        return response
    
    elif "missing" in query_lower or "null" in query_lower:
        missing_cols = {col: count for col, count in context['missing_values'].items() if count > 0}
        
        if not missing_cols:
            return "✅ Great news! Your dataset has no missing values."
        
        response = f"**Missing Value Analysis:**\n\n"
        response += f"Found missing values in {len(missing_cols)} columns:\n\n"
        
        for col, count in sorted(missing_cols.items(), key=lambda x: x[1], reverse=True)[:5]:
            pct = (count / context['shape'][0]) * 100
            response += f"• **{col}**: {count:,} missing ({pct:.1f}%)\n"
        
        response += "\n**Recommendations:**\n"
        for col, count in list(missing_cols.items())[:3]:
            pct = (count / context['shape'][0]) * 100
            if pct > 50:
                response += f"- Consider dropping '{col}' (>{pct:.0f}% missing)\n"
            elif pct > 20:
                response += f"- Use advanced imputation for '{col}'\n"
            else:
                response += f"- Simple imputation works for '{col}'\n"
        
        return response
    
    elif "correlation" in query_lower or "relationship" in query_lower:
        if len(context['numeric_columns']) < 2:
            return "You need at least 2 numeric columns to analyze correlations."
        
        # Calculate correlations for numeric columns
        numeric_df = df[context['numeric_columns']]
        corr_matrix = numeric_df.corr()
        
        # Find high correlations
        high_corr = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:
                    high_corr.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_val))
        
        response = "**Correlation Analysis:**\n\n"
        if high_corr:
            response += f"Found {len(high_corr)} strong correlations (>0.7):\n\n"
            for col1, col2, corr in sorted(high_corr, key=lambda x: abs(x[2]), reverse=True)[:5]:
                response += f"• **{col1}** ↔ **{col2}**: {corr:.3f}\n"
            response += "\n**Recommendation:** Consider removing highly correlated features to avoid multicollinearity."
        else:
            response += "No strong correlations (>0.7) found between numeric features."
        
        return response
    
    elif "quality" in query_lower or "issue" in query_lower:
        issues = []
        
        # Check for duplicates
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            issues.append(f"🔴 {duplicates:,} duplicate records ({duplicates/len(df)*100:.1f}%)")
        
        # Check for missing values
        missing_cols = sum(1 for count in context['missing_values'].values() if count > 0)
        if missing_cols > 0:
            issues.append(f"🟡 {missing_cols} columns with missing values")
        
        # Check for constant columns
        constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
        if constant_cols:
            issues.append(f"🟡 {len(constant_cols)} constant columns")
        
        # Check for high cardinality categorical columns
        high_card_cols = [col for col in context['categorical_columns'] 
                         if df[col].nunique() > len(df) * 0.8]
        if high_card_cols:
            issues.append(f"🟡 {len(high_card_cols)} high cardinality categorical columns")
        
        response = "**Data Quality Assessment:**\n\n"
        if issues:
            response += "**Issues Found:**\n"
            for issue in issues:
                response += f"{issue}\n"
            
            response += "\n**Priority Actions:**\n"
            response += "1. Remove duplicates first\n"
            response += "2. Handle missing values in important columns\n"
            response += "3. Remove or encode constant columns\n"
            response += "4. Consider feature engineering for high cardinality columns\n"
        else:
            response += "✅ **Excellent!** No major data quality issues detected.\n"
            response += "Your dataset appears to be well-structured and clean."
        
        return response
    
    elif "visualization" in query_lower or "chart" in query_lower or "plot" in query_lower:
        response = "**Visualization Recommendations:**\n\n"
        
        if context['numeric_columns']:
            response += "**For Numeric Data:**\n"
            response += "• Histograms: Show distribution of individual variables\n"
            response += "• Box plots: Identify outliers and compare distributions\n"
            response += "• Scatter plots: Explore relationships between variables\n"
            if len(context['numeric_columns']) > 1:
                response += "• Correlation heatmap: View all correlations at once\n"
        
        if context['categorical_columns']:
            response += "\n**For Categorical Data:**\n"
            response += "• Bar charts: Show frequency of categories\n"
            response += "• Grouped charts: Compare categories across groups\n"
        
        if context['numeric_columns'] and context['categorical_columns']:
            response += "\n**For Mixed Data:**\n"
            response += "• Box plots by category: Compare numeric distributions across groups\n"
            response += "• Colored scatter plots: Add categorical dimension to numeric relationships\n"
        
        return response
    
    elif "preprocessing" in query_lower or "prepare" in query_lower:
        steps = []
        
        # Missing values
        missing_cols = sum(1 for count in context['missing_values'].values() if count > 0)
        if missing_cols > 0:
            steps.append("1. **Handle Missing Values**: Use imputation or removal based on percentage missing")
        
        # Duplicates
        if df.duplicated().sum() > 0:
            steps.append("2. **Remove Duplicates**: Clean duplicate records")
        
        # Data types
        steps.append("3. **Optimize Data Types**: Convert strings to categories, ensure numeric types are correct")
        
        # Outliers
        if context['numeric_columns']:
            steps.append("4. **Handle Outliers**: Detect and treat outliers in numeric columns")
        
        # Encoding
        if context['categorical_columns']:
            steps.append("5. **Encode Categoricals**: Use one-hot encoding or target encoding")
        
        # Scaling
        if len(context['numeric_columns']) > 1:
            steps.append("6. **Feature Scaling**: Standardize or normalize numeric features")
        
        response = "**Preprocessing Recommendations:**\n\n"
        response += "\n".join(steps)
        
        response += "\n\n**Next Steps:**\n"
        response += "• Start with data cleaning (steps 1-2)\n"
        response += "• Then optimize data types and handle outliers\n"
        response += "• Finally apply encoding and scaling before modeling\n"
        
        return response
    
    else:
        # Generic response
        return f"""I'd be happy to help you analyze your dataset! 

**Your Dataset Summary:**
- **Size**: {context['shape'][0]:,} rows × {context['shape'][1]} columns
- **Data Types**: {len(context['numeric_columns'])} numeric, {len(context['categorical_columns'])} categorical
- **Completeness**: {len([col for col, count in context['missing_values'].items() if count == 0])} complete columns

**What I can help you with:**
• Data quality assessment and cleaning recommendations
• Missing value analysis and treatment strategies  
• Correlation analysis and feature relationships
• Visualization suggestions for your specific data types
• Preprocessing steps for machine learning
• Outlier detection and treatment options

Feel free to ask me specific questions about any aspect of your data!"""

# ================================
# Upload Dataset
# ================================
st.header("1. Upload Dataset")
uploaded_file = st.file_uploader(
    "Choose a file", 
    type=['csv', 'xlsx', 'xls', 'json', 'parquet'],
    help="Supported formats: CSV, Excel, JSON, Parquet"
)

if uploaded_file:
    # Load data only once and cache it
    if "df" not in st.session_state or st.session_state.get('current_file') != uploaded_file.name:
        with st.spinner("Loading dataset..."):
            df = load_data(uploaded_file)
            if df is not None:
                update_df(df)
                st.session_state.original_df = df.copy()
                st.session_state.current_file = uploaded_file.name
                st.success(f"✅ Successfully loaded {uploaded_file.name}")
            else:
                st.stop()

df = st.session_state.get('df')

# ⬅️ ADDED: Initialize history and redo stack if not already
if 'df' not in st.session_state:
    st.session_state.df = None
if 'history' not in st.session_state:
    st.session_state.history = []
if 'redo_stack' not in st.session_state:
    st.session_state.redo_stack = []

# ⬅️ ADDED: Central function to update df with history tracking
def update_df(new_df):
    """Update the dataframe while keeping history for undo/redo."""
    if st.session_state.df is not None:
        st.session_state.history.append(st.session_state.df.copy())  # save current state
    st.session_state.redo_stack = []  # clear redo stack on new action
    st.session_state.df = new_df

# ⬅️ ADDED: Undo / Redo buttons
undo_col, redo_col = st.columns([1, 1])
with undo_col:
    if st.button("↩️ Undo"):
        if st.session_state.history:
            st.session_state.redo_stack.append(st.session_state.df.copy())
            st.session_state.df = st.session_state.history.pop()
        else:
            st.info("Nothing to undo")

with redo_col:
    if st.button("↪️ Redo"):
        if st.session_state.redo_stack:
            st.session_state.history.append(st.session_state.df.copy())
            st.session_state.df = st.session_state.redo_stack.pop()
        else:
            st.info("Nothing to redo")

if df is not None:
    # Get column types
    col_types = get_column_types(df)
    
    # ================================
    # Dataset Overview
    # ================================
    st.header("2. Dataset Overview")
    
    # Metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Rows", f"{len(df):,}")
    with col2:
        st.metric("Columns", len(df.columns))
    with col3:
        memory_mb = df.memory_usage(deep=True).sum() / (1024**2)
        st.metric("Memory (MB)", f"{memory_mb:.2f}")
    with col4:
        duplicates = df.duplicated().sum()
        st.metric("Duplicate Rows", f"{duplicates:,}")
    with col5:
        missing_cells = df.isnull().sum().sum()
        total_cells = len(df) * len(df.columns)
        missing_pct = (missing_cells / total_cells) * 100 if total_cells > 0 else 0
        st.metric("Missing %", f"{missing_pct:.1f}%")

    def render_column_info(df):
        """Enhanced column information display"""
        info_data = []
        for col in df.columns:
            col_data = {
                'Column': col,
                'Data Type': str(df[col].dtype),
                'Non-Null': f"{df[col].count():,}",
                'Null Count': f"{df[col].isnull().sum():,}",
                'Null %': f"{(df[col].isnull().sum()/len(df)*100):.1f}%",
                'Unique Values': f"{df[col].nunique():,}",
                'Memory (KB)': f"{df[col].memory_usage(deep=True)/1024:.1f}"
            }
            
            # Add sample values for categorical columns
            if col in col_types['categorical'] and df[col].nunique() <= 10:
                unique_vals = df[col].dropna().unique()[:5]
                col_data['Sample Values'] = ', '.join(map(str, unique_vals))
            else:
                col_data['Sample Values'] = ''
                
            info_data.append(col_data)
        
        info_df = pd.DataFrame(info_data)
        st.subheader("📋 Column Information")
        st.dataframe(info_df, use_container_width=True, hide_index=True)
        return info_df

    col_info = render_column_info(df)

    # Data Quality Summary
    st.subheader("🔍 Data Quality Summary")
    quality_col1, quality_col2 = st.columns(2)
    
    with quality_col1:
        st.write("**Column Type Distribution:**")
        type_counts = {
            'Numeric': len(col_types['numeric']),
            'Categorical': len(col_types['categorical']),
            'DateTime': len(col_types['datetime']),
            'Boolean': len(col_types['boolean'])
        }
        for dtype, count in type_counts.items():
            if count > 0:
                st.write(f"• {dtype}: {count} columns")
    
    with quality_col2:
        st.write("**Data Quality Issues:**")
        issues = []
        if duplicates > 0:
            issues.append(f"• {duplicates:,} duplicate rows ({duplicates/len(df)*100:.1f}%)")
        
        high_missing_cols = df.columns[df.isnull().sum() > len(df) * 0.5].tolist()
        if high_missing_cols:
            issues.append(f"• {len(high_missing_cols)} columns with >50% missing values")
        
        constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
        if constant_cols:
            issues.append(f"• {len(constant_cols)} constant/near-constant columns")
        
        if not issues:
            st.success("✅ No major quality issues detected")
        else:
            for issue in issues:
                st.warning(issue)

    # ================================
    # View Dataset
    # ================================
    st.header("3. 👀 View Dataset")
    view_col1, view_col2 = st.columns([1, 3])
    
    with view_col1:
        # Limit slider max
        max_rows = min(len(df), 1000)
        num_rows = st.slider("Number of rows to display", 1, max_rows, min(10, len(df)))
        view_option = st.selectbox("View Option", ["Top Rows", "Bottom Rows", "Random Sample"])

        # Sorting options
        sort_column = st.selectbox("Sort by column", options=[None] + list(df.columns))  # None = no sort
        sort_order = st.radio("Sort order", ["Ascending", "Descending"])

    with view_col2:
        if st.button("📄 View Dataset", type="primary"):
            try:
                # Start with actual dataset
                display_df = df.copy()

                # Apply sorting to actual dataset (updates state)
                if sort_column and sort_column in df.columns:
                    ascending = True if sort_order == "Ascending" else False
                    sorted_df = df.sort_values(by=sort_column, ascending=ascending).reset_index(drop=True)
                    update_df(sorted_df)  # 🔥 Save sorted dataset
                    display_df = sorted_df.copy()
                else:
                    display_df = df.copy()

                # Slice rows based on view option
                if view_option == "Top Rows":
                    display_df = display_df.head(num_rows)
                elif view_option == "Bottom Rows":
                    display_df = display_df.tail(num_rows)
                else:  # Random Sample
                    display_df = display_df.sample(n=min(num_rows, len(display_df)), random_state=42)

                # Show sorted + sliced dataframe
                st.dataframe(display_df, use_container_width=True, hide_index=True)

            except Exception as e:
                st.error(f"Error displaying data: {str(e)}")

        	

    # Quick Statistics
    st.subheader("📈 Quick Statistics")
    
    if col_types['numeric']:
        stats_col = st.selectbox("Select column for detailed statistics", col_types['numeric'])
        
        if stats_col:
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Basic Statistics:**")
                stats = df[stats_col].describe()
                for stat, value in stats.items():
                    st.write(f"• {stat.title()}: {value:.2f}")
            
            with col2:
                st.write("**Additional Info:**")
                st.write(f"• Skewness: {df[stats_col].skew():.2f}")
                st.write(f"• Kurtosis: {df[stats_col].kurtosis():.2f}")
                st.write(f"• Missing Values: {df[stats_col].isnull().sum():,}")
                
                outlier_info = detect_outliers(df, stats_col)
                if outlier_info:
                    outlier_count = outlier_info[0]
                    st.write(f"• Outliers (IQR method): {outlier_count:,}")



    # ================================
    # Data Restructuring (NEW FEATURE)
    # ================================
    st.header("4. 🔄 Data Restructuring")
    
    restructure_tabs = st.tabs(["Pivot Data", "Unpivot Data"])
    
    with restructure_tabs[0]:  # Pivot
        st.subheader("Pivot Table Creation")
        st.info("Create pivot tables to reshape your data from long to wide format")
        
        pivot_col1, pivot_col2, pivot_col3 = st.columns(3)
        
        with pivot_col1:
            index_cols = st.multiselect("Index Columns (rows)", df.columns, key="pivot_index")
            
        with pivot_col2:
            columns_col = st.selectbox("Columns (pivot)", df.columns, key="pivot_columns")
            
        with pivot_col3:
            values_col = st.selectbox("Values (aggregated)", df.columns, key="pivot_values")
            aggfunc = st.selectbox("Aggregation Function", ["sum", "mean", "count", "min", "max", "std"])
        
        if st.button("🔄 Create Pivot Table", type="primary"):
            if index_cols and columns_col and values_col:
                try:
                    pivot_df = df.pivot_table(
                        index=index_cols, 
                        columns=columns_col, 
                        values=values_col, 
                        aggfunc=aggfunc, 
                        fill_value=0
                    )
                    
                    # Reset index to make it a regular dataframe
                    pivot_df = pivot_df.reset_index()
                    
                    st.success(f"✅ Pivot table created! Shape: {pivot_df.shape}")
                    st.dataframe(pivot_df, use_container_width=True)
                    
                    # Option to replace main dataframe
                    if st.button("Replace Main Dataset with Pivot", type="secondary"):
                        st.session_state.df = pivot_df
                        st.rerun()
                    
                    # Download pivot table
                    csv_data = pivot_df.to_csv(index=False)
                    st.download_button(
                        "📥 Download Pivot Table", 
                        csv_data, 
                        "pivot_table.csv", 
                        "text/csv"
                    )
                    
                except Exception as e:
                    st.error(f"Error creating pivot table: {str(e)}")
            else:
                st.warning("Please select all required columns for pivot operation")

    with restructure_tabs[1]:  # Unpivot
        st.subheader("Unpivot Data (Melt)")
        st.info("Transform your data from wide to long format")
        
        unpivot_col1, unpivot_col2 = st.columns(2)
        
        with unpivot_col1:
            id_vars = st.multiselect("ID Variables (keep as columns)", df.columns, key="unpivot_id")
            value_vars = st.multiselect("Value Variables (columns to unpivot)", df.columns, key="unpivot_value")
            
        with unpivot_col2:
            var_name = st.text_input("Variable Column Name", value="variable")
            value_name = st.text_input("Value Column Name", value="value")
        
        if st.button("🔄 Unpivot Data", type="primary"):
            try:
                # If no value_vars specified, use all columns except id_vars
                if not value_vars:
                    value_vars = [col for col in df.columns if col not in id_vars]
                
                unpivot_df = pd.melt(
                    df, 
                    id_vars=id_vars if id_vars else None,
                    value_vars=value_vars,
                    var_name=var_name,
                    value_name=value_name
                )
                
                st.success(f"✅ Data unpivoted! Shape: {unpivot_df.shape}")
                st.dataframe(unpivot_df.head(100), use_container_width=True)
                
                # Option to replace main dataframe
                if st.button("Replace Main Dataset with Unpivoted Data", type="secondary"):
                    st.session_state.df = unpivot_df
                    st.rerun()
                
                # Download unpivoted data
                csv_data = unpivot_df.to_csv(index=False)
                st.download_button(
                    "📥 Download Unpivoted Data", 
                    csv_data, 
                    "unpivoted_data.csv", 
                    "text/csv"
                )
                
            except Exception as e:
                st.error(f"Error unpivoting data: {str(e)}")

    # ================================
    # Data Transformation
    # ================================
    st.header("5. 🔧 Data Transformation")
    
    transform_tabs = st.tabs(["Remove Columns", "Change Data Types", "Rename Columns"])
    
    with transform_tabs[0]:
        st.subheader("Remove Columns")
        drop_cols = st.multiselect(
            "Select columns to drop", 
            df.columns,
            help="Select multiple columns to remove from the dataset"
        )
        if st.button("🗑 Drop Selected Columns", type="primary"):
            if drop_cols:
                try:
                    df = df.drop(columns=drop_cols)
                    update_df(df)
                    st.success(f"✅ Successfully dropped columns: {', '.join(drop_cols)}")
                    col_info = render_column_info(df)
                    col_types = get_column_types(df)  # Update column types
                except Exception as e:
                    st.error(f"Error dropping columns: {str(e)}")
            else:
                st.warning("Please select at least one column to drop")

    with transform_tabs[1]:
        st.subheader("Change Column Data Types")
        trans_col1, trans_col2 = st.columns(2)
        
        with trans_col1:
            col_to_change = st.selectbox("Select column", df.columns if len(df.columns) > 0 else [])
            
        with trans_col2:
            new_dtype = st.selectbox(
                "New data type", 
                ["object", "int", "float", "datetime", "category", "boolean"]
            )
        
        if col_to_change and st.button("🔄 Convert Data Type", type="primary"):
            try:
                original_dtype = df[col_to_change].dtype
                
                if new_dtype == "datetime":
                    df[col_to_change] = pd.to_datetime(df[col_to_change], errors="coerce")
                elif new_dtype == "int":
                    df[col_to_change] = pd.to_numeric(df[col_to_change], errors="coerce").astype("Int64")
                elif new_dtype == "float":
                    df[col_to_change] = pd.to_numeric(df[col_to_change], errors="coerce").astype(float)
                elif new_dtype == "category":
                    df[col_to_change] = df[col_to_change].astype("category")
                elif new_dtype == "boolean":
                    df[col_to_change] = df[col_to_change].astype(bool)
                else:
                    df[col_to_change] = df[col_to_change].astype(str)
                
                update_df(df)
                st.success(f"✅ Successfully converted '{col_to_change}' from {original_dtype} to {df[col_to_change].dtype}")
                col_types = get_column_types(df)  # Update column types
                
            except Exception as e:
                st.error(f"❌ Error converting data type: {str(e)}")

    with transform_tabs[2]:
        st.subheader("Rename Columns")
        rename_col1, rename_col2 = st.columns(2)
        
        with rename_col1:
            rename_col = st.selectbox("Select column to rename", df.columns if len(df.columns) > 0 else [])
            
        with rename_col2:
            new_name = st.text_input("New column name", value="")
        
        if rename_col and new_name and st.button("✅ Rename Column", type="primary"):
            if new_name in df.columns:
                st.error(f"❌ Column name '{new_name}' already exists")
            else:
                try:
                    df = df.rename(columns={rename_col: new_name})
                    update_df(df)
                    st.success(f"✅ Successfully renamed '{rename_col}' to '{new_name}'")
                    col_types = get_column_types(df)  # Update column types
                except Exception as e:
                    st.error(f"❌ Error renaming column: {str(e)}")

    # ================================
    # Handle Data Issues
    # ================================
    st.header("6. 🔧 Handle Data Issues")

    tab_missing, tab_duplicates, tab_outliers, tab_cleaning, tab_conditional = st.tabs(
        ["Missing Values", "Duplicate Records", "Outliers", "Feature Cleaning", "Conditional Column"]
    )

    with tab_missing:  # Missing Values
        st.subheader("Missing Value Treatment")

        # Show missing value summary
        missing_summary = df.isnull().sum()
        missing_cols = missing_summary[missing_summary > 0].index.tolist()

        if not missing_cols:
            st.success("✅ No missing values found in the dataset")
        else:
            st.write("**Columns with missing values:**")
            for col in missing_cols:
                missing_count = missing_summary[col]
                missing_pct = (missing_count / len(df)) * 100
                st.write(f"• **{col}**: {missing_count:,} missing ({missing_pct:.1f}%)")

            miss_col1, miss_col2 = st.columns(2)

            with miss_col1:
                issue_col = st.selectbox("Select column", missing_cols)

            with miss_col2:
                action = st.selectbox("Action", ["Remove Rows", "Impute"])

            if action == "Impute" and issue_col:
                if issue_col in col_types['numeric']:
                    impute_options = ["Mean", "Median", "Mode", "Forward Fill", "Backward Fill"]
                else:
                    impute_options = ["Mode", "Forward Fill", "Backward Fill", "Constant Value"]

                impute_type = st.selectbox("Impute method", impute_options)

                if impute_type == "Constant Value":
                    const_value = st.text_input("Enter constant value")

            if st.button("✅ Apply Missing Value Fix", type="primary"):
                try:
                    original_len = len(df)

                    if action == "Remove Rows":
                        df = df.dropna(subset=[issue_col])
                        removed_rows = original_len - len(df)
                        st.success(f"✅ Removed {removed_rows:,} rows with missing values in '{issue_col}'")

                    elif action == "Impute":
                        if impute_type == "Mean" and issue_col in col_types['numeric']:
                            df[issue_col].fillna(df[issue_col].mean(), inplace=True)
                        elif impute_type == "Median" and issue_col in col_types['numeric']:
                            df[issue_col].fillna(df[issue_col].median(), inplace=True)
                        elif impute_type == "Mode":
                            mode_val = df[issue_col].mode()[0] if not df[issue_col].mode().empty else 'Unknown'
                            df[issue_col].fillna(mode_val, inplace=True)
                        elif impute_type == "Forward Fill":
                            df[issue_col].fillna(method='ffill', inplace=True)
                        elif impute_type == "Backward Fill":
                            df[issue_col].fillna(method='bfill', inplace=True)
                        elif impute_type == "Constant Value" and 'const_value' in locals():
                            df[issue_col].fillna(const_value, inplace=True)

                        st.success(f"✅ Imputed missing values in '{issue_col}' using {impute_type}")

                    update_df(df)

                except Exception as e:
                    st.error(f"❌ Error handling missing values: {str(e)}")

    with tab_duplicates:  # Duplicates (ENHANCED)
        st.subheader("Duplicate Record Treatment")

        duplicates = df.duplicated().sum()
        if duplicates == 0:
            st.success("✅ No duplicate records found")
        else:
            st.warning(f"⚠️ Found {duplicates:,} duplicate records ({duplicates/len(df)*100:.1f}% of data)")

            # View duplicate records and actions
            view_duplicates_col, action_col = st.columns(2)

            with view_duplicates_col:
                if st.button("👀 View Duplicate Records", type="secondary"):
                    duplicate_mask = df.duplicated(keep=False)
                    duplicate_records = df[duplicate_mask].sort_values(by=df.columns.tolist())

                    if len(duplicate_records) > 0:
                        st.write(f"**Showing all {len(duplicate_records)} duplicate records:**")
                        st.dataframe(duplicate_records, use_container_width=True)

                        st.write("**Duplicate Groups Analysis:**")
                        duplicate_groups = duplicate_records.groupby(df.columns.tolist()).size().reset_index(name='count')
                        duplicate_groups = duplicate_groups[duplicate_groups['count'] > 1].sort_values('count', ascending=False)

                        if len(duplicate_groups) > 0:
                            st.write("Groups with multiple identical records:")
                            st.dataframe(duplicate_groups, use_container_width=True)

                        csv_data = duplicate_records.to_csv(index=True)
                        st.download_button(
                            "📥 Download Duplicate Records",
                            csv_data,
                            "duplicate_records.csv",
                            "text/csv"
                        )

            with action_col:
                dup_method = st.selectbox(
                    "Duplicate handling method",
                    ["Remove all duplicates", "Keep first occurrence", "Keep last occurrence", "Mark duplicates (add column)"]
                )

                if st.button("🧹 Handle Duplicates", type="primary"):
                    try:
                        original_len = len(df)

                        if dup_method == "Remove all duplicates":
                            df = df.drop_duplicates(keep=False)
                        elif dup_method == "Keep first occurrence":
                            df = df.drop_duplicates(keep='first')
                        elif dup_method == "Keep last occurrence":
                            df = df.drop_duplicates(keep='last')
                        elif dup_method == "Mark duplicates (add column)":
                            df['is_duplicate'] = df.duplicated(keep=False)
                            st.success(f"✅ Added 'is_duplicate' column marking {df['is_duplicate'].sum()} duplicate records")

                        removed = original_len - len(df)
                        update_df(df)
                        st.success(f"✅ Removed {removed:,} duplicate records")

                    except Exception as e:
                        st.error(f"❌ Error handling duplicates: {str(e)}")

    with tab_outliers:  # Outliers
        st.subheader("Outlier Treatment")

        numeric_cols = col_types['numeric']
        if not numeric_cols:
            st.info("No numeric columns found for outlier detection")
        else:
            outlier_col = st.selectbox("Select numeric column", numeric_cols)

            if outlier_col:
                outlier_info = detect_outliers(df, outlier_col)
                if outlier_info:
                    outlier_count, lower_bound, upper_bound = outlier_info

                    if outlier_count == 0:
                        st.success(f"✅ No outliers detected in '{outlier_col}'")
                    else:
                        st.warning(f"⚠️ Found {outlier_count:,} outliers in '{outlier_col}' ({outlier_count/len(df)*100:.1f}% of data)")
                        st.write(f"**Outlier bounds:** [{lower_bound:.2f}, {upper_bound:.2f}]")

                        method = st.selectbox("Treatment method", ["Remove outliers", "Cap to bounds", "Winsorize"])

                        if st.button("🔧 Apply Outlier Treatment", type="primary"):
                            try:
                                original_len = len(df)

                                if method == "Remove outliers":
                                    df = df[(df[outlier_col] >= lower_bound) & (df[outlier_col] <= upper_bound)]
                                    removed = original_len - len(df)
                                    st.success(f"✅ Removed {removed:,} outlier records")

                                elif method == "Cap to bounds":
                                    df[outlier_col] = np.clip(df[outlier_col], lower_bound, upper_bound)
                                    st.success(f"✅ Capped outliers in '{outlier_col}' to bounds")

                                elif method == "Winsorize":
                                    from scipy.stats import mstats
                                    df[outlier_col] = mstats.winsorize(df[outlier_col], limits=[0.05, 0.05])
                                    st.success(f"✅ Applied winsorization to '{outlier_col}'")

                                update_df(df)

                            except Exception as e:
                                st.error(f"❌ Error treating outliers: {str(e)}")

    with tab_cleaning:  # Feature Cleaning
        st.subheader("Feature Cleaning")

        clean_col1, clean_col2 = st.columns(2)

        with clean_col1:
            if st.button("🧹 Remove Constant Columns", type="secondary"):
                constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
                if constant_cols:
                    df = df.drop(columns=constant_cols)
                    update_df(df)
                    st.success(f"✅ Removed {len(constant_cols)} constant columns: {', '.join(constant_cols)}")
                else:
                    st.info("No constant columns found")

        with clean_col2:
            if st.button("🔍 Remove High Correlation", type="secondary"):
                numeric_df = df.select_dtypes(include=[np.number])
                if len(numeric_df.columns) > 1:
                    corr_matrix = numeric_df.corr().abs()
                    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                    high_corr = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]

                    if high_corr:
                        df = df.drop(columns=high_corr)
                        update_df(df)
                        st.success(f"✅ Removed {len(high_corr)} highly correlated columns")
                    else:
                        st.info("No highly correlated columns found (>0.95)")
                else:
                    st.info("Need at least 2 numeric columns for correlation analysis")

    with tab_conditional:  # Conditional Column
        st.subheader("Conditional Column")

        col1 = st.selectbox("Select first column", df.columns, key="cond_col1")

        use_constant = st.checkbox("Use a constant value instead of second column?", key="cond_use_constant")
        if use_constant:
            value2 = st.text_input("Enter constant value", key="cond_value2")
            try:
                value2 = float(value2)
            except:
                pass
        else:
            col2 = st.selectbox("Select second column", df.columns, key="cond_col2")
            value2 = df[col2]

        condition = st.selectbox("Select condition", [">", "<", ">=", "<=", "==", "!="], key="cond_operator")

        # Build mask safely
        try:
            import operator
            ops = {">": operator.gt, "<": operator.lt, ">=": operator.ge, "<=": operator.le, "==": operator.eq, "!=": operator.ne}
            op_func = ops[condition]
            mask = op_func(df[col1], value2)
            violating_rows = df[mask]

            # Show results
            if st.checkbox("👀 View matching rows", key="cond_view"):
                if violating_rows.empty:
                    st.info("No rows match the condition.")
                else:
                    st.dataframe(violating_rows, use_container_width=True)

            # Action choice
            replace_mode = st.radio("Choose action", ["Remove rows", "Replace values"], key="cond_action")

            # 🔹 Move replacement input outside Apply button
            replacement = None
            if replace_mode == "Replace values":
                if pd.api.types.is_numeric_dtype(df[col1]):
                    replacement = st.number_input("Enter replacement value", key="cond_replace_value_num")
                else:
                    replacement = st.text_input("Enter replacement value", key="cond_replace_value_txt")

            if st.button("✅ Apply", type="primary", key="cond_apply"):
                if violating_rows.empty:
                    st.success("✅ No rows match the condition")
                else:
                    if replace_mode == "Remove rows":
                        new_df = df[~mask]
                        update_df(new_df)
                        st.success(f"✅ Removed {len(violating_rows)} rows based on condition '{col1} {condition} {value2}'")

                    elif replace_mode == "Replace values":
                        if replacement not in [None, ""]:
                            new_df = df.copy()
                            new_df.loc[mask, col1] = replacement
                            update_df(new_df)
                            st.success(f"✅ Replaced {len(violating_rows)} values in '{col1}' where '{col1} {condition} {value2}'")
                        else:
                            st.warning("⚠️ Please enter a replacement value before applying")

                        if replacement != "":
                            new_df = df.copy()
                            new_df.loc[mask, col1] = replacement
                            update_df(new_df)
                            st.success(f"✅ Replaced {len(violating_rows)} values in '{col1}' where '{col1} {condition} {value2}'")

        except Exception as e:
            st.error(f"❌ Error applying conditional operation: {str(e)}")



    # ================================
    # Enhanced Visualizations
    # ================================
    st.header("7. 📊 Data Visualizations")
    
    # Update column types after any transformations
    col_types = get_column_types(df)
    
    viz_col1, viz_col2 = st.columns([2, 1])
    
    with viz_col1:
        chart_type = st.selectbox(
            "Select chart type", 
            ["Histogram", "Box Plot", "Bar Chart", "Scatter Plot", "Correlation Heatmap", "Distribution Plot"]
        )
    
    with viz_col2:
        if chart_type in ["Histogram", "Box Plot", "Distribution Plot"]:
            available_cols = col_types['numeric']
            if not available_cols:
                st.warning("No numeric columns available for this chart type")
                st.stop()
        elif chart_type == "Bar Chart":
            available_cols = col_types['categorical'] + col_types['boolean']
            if not available_cols:
                st.warning("No categorical columns available for bar chart")
                st.stop()
        elif chart_type == "Scatter Plot":
            available_cols = col_types['numeric']
            if len(available_cols) < 2:
                st.warning("Need at least 2 numeric columns for scatter plot")
                st.stop()
        elif chart_type == "Correlation Heatmap":
            available_cols = col_types['numeric']
            if len(available_cols) < 2:
                st.warning("Need at least 2 numeric columns for correlation heatmap")
                st.stop()
        else:
            available_cols = list(df.columns)
    
    # Column selection based on chart type
    viz_setup_col1, viz_setup_col2, viz_setup_col3 = st.columns(3)
    
    with viz_setup_col1:
        if chart_type == "Scatter Plot":
            x_col = st.selectbox("Select X-axis column", available_cols)
            y_col = st.selectbox("Select Y-axis column", [col for col in available_cols if col != x_col])
        elif chart_type == "Correlation Heatmap":
            selected_cols = st.multiselect("Select columns (leave empty for all numeric)", available_cols)
        else:
            vis_col = st.selectbox("Select column to visualize", available_cols) if available_cols else None
    
    with viz_setup_col2:
        # Hue selection - categorical columns for grouping
        hue_options = col_types['categorical'] + col_types['boolean']
        if hue_options and chart_type != "Correlation Heatmap":
            hue_col = st.selectbox("Select grouping variable (hue)", [None] + hue_options)
        else:
            hue_col = None
    
    with viz_setup_col3:
        # Additional options based on chart type
        if chart_type == "Histogram":
            bins = st.slider("Number of bins", 10, 100, 30)
        elif chart_type in ["Box Plot", "Bar Chart"]:
            show_values = st.checkbox("Show values on plot", value=False)
    
    # Generate visualization
    if st.button("📊 Generate Visualization", type="primary"):
        try:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            if chart_type == "Histogram" and vis_col:
                if hue_col:
                    # Create histogram with hue
                    for category in df[hue_col].dropna().unique():
                        subset = df[df[hue_col] == category]
                        ax.hist(subset[vis_col].dropna(), bins=bins, alpha=0.7, label=str(category), edgecolor='black')
                    ax.legend(title=hue_col)
                else:
                    ax.hist(df[vis_col].dropna(), bins=bins, edgecolor='black', alpha=0.7)
                ax.set_xlabel(vis_col)
                ax.set_ylabel('Frequency')
                ax.set_title(f'Histogram of {vis_col}')
            
            elif chart_type == "Box Plot" and vis_col:
                if hue_col:
                    sns.boxplot(data=df, y=vis_col, x=hue_col, ax=ax)
                    plt.xticks(rotation=45)
                else:
                    sns.boxplot(data=df, y=vis_col, ax=ax)
                ax.set_title(f'Box Plot of {vis_col}')
            
            elif chart_type == "Bar Chart" and vis_col:
                if hue_col and hue_col != vis_col:
                    # Create grouped bar chart
                    cross_tab = pd.crosstab(df[vis_col], df[hue_col])
                    cross_tab.plot(kind='bar', ax=ax)
                    ax.legend(title=hue_col, bbox_to_anchor=(1.05, 1), loc='upper left')
                else:
                    # Simple bar chart
                    value_counts = df[vis_col].value_counts().head(20)  # Limit to top 20
                    value_counts.plot(kind='bar', ax=ax)
                    if show_values:
                        for i, v in enumerate(value_counts.values):
                            ax.text(i, v + max(value_counts.values) * 0.01, str(v), ha='center', va='bottom')
                plt.xticks(rotation=45)
                ax.set_title(f'Bar Chart of {vis_col}')
            
            elif chart_type == "Scatter Plot":
                if hue_col:
                    for category in df[hue_col].dropna().unique():
                        subset = df[df[hue_col] == category]
                        ax.scatter(subset[x_col], subset[y_col], label=str(category), alpha=0.6)
                    ax.legend(title=hue_col, bbox_to_anchor=(1.05, 1), loc='upper left')
                else:
                    ax.scatter(df[x_col], df[y_col], alpha=0.6)
                ax.set_xlabel(x_col)
                ax.set_ylabel(y_col)
                ax.set_title(f'Scatter Plot: {x_col} vs {y_col}')
            
            elif chart_type == "Distribution Plot" and vis_col:
                if hue_col:
                    for category in df[hue_col].dropna().unique():
                        subset = df[df[hue_col] == category]
                        sns.kdeplot(data=subset[vis_col].dropna(), label=str(category), ax=ax)
                    ax.legend(title=hue_col)
                else:
                    sns.kdeplot(data=df[vis_col].dropna(), ax=ax)
                ax.set_xlabel(vis_col)
                ax.set_title(f'Distribution Plot of {vis_col}')
            
            elif chart_type == "Correlation Heatmap":
                cols_to_use = selected_cols if selected_cols else available_cols
                corr_data = df[cols_to_use].corr()
                sns.heatmap(corr_data, annot=True, cmap='coolwarm', center=0, 
                           square=True, linewidths=0.5, ax=ax, fmt='.2f')
                ax.set_title('Correlation Heatmap')
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Download option
            buf = BytesIO()
            fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
            buf.seek(0)
            st.download_button(
                "💾 Download Chart as PNG", 
                buf, 
                file_name=f"{chart_type.lower().replace(' ', '_')}_{vis_col if 'vis_col' in locals() else 'chart'}.png",
                mime="image/png"
            )
            
            plt.close(fig)  # Close figure to free memory
            
        except Exception as e:
            st.error(f"❌ Error creating visualization: {str(e)}")
            st.info("💡 Tip: Try selecting different columns or check for data issues")


    # ================================
    # Advanced Machine Learning Pipeline
    # ================================
    st.header("8. 🤖 Advanced Machine Learning Pipeline")
    
    if len(df) < 20:
        st.error("⚠️ Dataset too small for advanced modeling (minimum 20 rows required)")
    else:
        ml_tabs = st.tabs([
            "🎯 Setup", 
            "⚖️ Handle Imbalance", 
            "🔧 Feature Engineering",
            "🏋️ Model Training",
            "📈 Evaluation", 
            "🔍 Interpretability",
            "🔗 Ensemble Models",
            "💾 Model Management"
        ])
        
        with ml_tabs[0]:  # Setup
            st.subheader("🎯 Machine Learning Setup")
            
            setup_col1, setup_col2 = st.columns(2)
            
            with setup_col1:
                st.write("**Select Target Variable**")
                target_variable = st.selectbox("What do you want to predict?", df.columns)
                
                if target_variable:
                    target_info = analyze_target_variable(df, target_variable)
                    st.write(f"**Problem Type:** {target_info['type']}")
                    st.write(f"**Unique Values:** {target_info['unique_count']}")
                    
                    if target_info['type'] == 'Classification':
                        # Check for class imbalance
                        is_imbalanced, imbalance_info = check_class_imbalance(df[target_variable])
                        
                        if is_imbalanced:
                            st.warning(f"⚠️ **Class Imbalance Detected!**")
                            st.write(f"Imbalance Ratio: {imbalance_info['ratio']:.2f}:1")
                            st.write(f"Majority Class: {imbalance_info['majority_class']} ({imbalance_info['majority_count']} samples)")
                            st.write(f"Minority Class: {imbalance_info['minority_class']} ({imbalance_info['minority_count']} samples)")
                        else:
                            st.success("✅ Classes are relatively balanced")
                        
                        # Class distribution
                        st.write("**Class Distribution:**")
                        class_dist = df[target_variable].value_counts()
                        fig, ax = plt.subplots(figsize=(8, 5))
                        class_dist.plot(kind='bar', ax=ax)
                        ax.set_title('Target Variable Distribution')
                        ax.set_ylabel('Count')
                        plt.xticks(rotation=45)
                        st.pyplot(fig)
                        plt.close(fig)
                    
                    # Store in session state
                    st.session_state.ml_target = target_variable
                    st.session_state.ml_problem_type = target_info['type']
                    st.session_state.ml_is_imbalanced = is_imbalanced if target_info['type'] == 'Classification' else False
            
            with setup_col2:
                st.write("**Feature Selection**")
                available_features = [col for col in df.columns if col != target_variable]
                
                feature_method = st.radio("Feature selection method:", 
                                         ["All Features", "Manual Selection", "Auto Selection"])
                
                if feature_method == "All Features":
                    selected_features = available_features
                elif feature_method == "Manual Selection":
                    selected_features = st.multiselect("Choose features:", available_features,
                                                      default=available_features[:min(10, len(available_features))])
                else:  # Auto Selection
                    n_features = st.slider("Number of features to select:", 5, min(50, len(available_features)), 20)
                    if st.button("🔍 Auto Select Features"):
                        selected_features = advanced_feature_selection(
                            df[available_features], df[target_variable], n_features=n_features).columns.tolist()
                        st.session_state.ml_features = selected_features
                        st.write(f"Selected {len(selected_features)} features: {', '.join(selected_features[:5])}...")
                
                if 'selected_features' in locals():
                    st.session_state.ml_features = selected_features
                    st.success(f"✅ Selected {len(selected_features)} features for modeling")

        with ml_tabs[1]:  # Handle Imbalance
            st.subheader("⚖️ Handle Imbalanced Data")
            
            if 'ml_target' not in st.session_state:
                st.warning("⚠️ Please complete the setup first")
            elif not st.session_state.get('ml_is_imbalanced', False):
                st.info("📊 Your dataset appears to be balanced. This section is for imbalanced classification problems.")
            else:
                st.write("**Resampling Techniques**")
                
                resampling_col1, resampling_col2 = st.columns(2)
                
                with resampling_col1:
                    resampling_method = st.selectbox("Select resampling method:", [
                        "None",
                        "SMOTE (Oversampling)",
                        "ADASYN (Adaptive Oversampling)", 
                        "Random Undersampling",
                        "SMOTE + Tomek (Combined)"
                    ])
                    
                    use_class_weights = st.checkbox("Use Class Weights", value=True,
                                                   help="Make models pay more attention to minority class")
                
                with resampling_col2:
                    if resampling_method != "None" and IMBLEARN_AVAILABLE:
                        st.write("**Resampling Preview**")
                        
                        # Show current distribution
                        current_dist = df[st.session_state.ml_target].value_counts()
                        st.write("**Current Distribution:**")
                        for class_name, count in current_dist.items():
                            st.write(f"• {class_name}: {count}")
                        
                        if st.button("🔄 Apply Resampling Preview"):
                            try:
                                X_preview = df[st.session_state.ml_features]
                                y_preview = df[st.session_state.ml_target]
                                
                                # Prepare data for resampling (numeric only)
                                X_numeric = X_preview.select_dtypes(include=[np.number])
                                if len(X_numeric.columns) == 0:
                                    st.error("Need numeric features for resampling preview")
                                else:
                                    method_map = {
                                        "SMOTE (Oversampling)": "smote",
                                        "ADASYN (Adaptive Oversampling)": "adasyn",
                                        "Random Undersampling": "undersample", 
                                        "SMOTE + Tomek (Combined)": "smote_tomek"
                                    }
                                    
                                    X_resampled, y_resampled = handle_imbalanced_data(
                                        X_numeric, y_preview, method_map[resampling_method]
                                    )
                                    
                                    st.write("**After Resampling:**")
                                    new_dist = pd.Series(y_resampled).value_counts()
                                    for class_name, count in new_dist.items():
                                        st.write(f"• {class_name}: {count}")
                                    
                                    # Store resampling settings
                                    st.session_state.resampling_method = method_map[resampling_method]
                                    st.session_state.use_class_weights = use_class_weights
                            
                            except Exception as e:
                                st.error(f"Resampling preview failed: {str(e)}")
                    elif not IMBLEARN_AVAILABLE:
                        st.warning("⚠️ Install imbalanced-learn package for resampling features")
                
                # Store settings
                st.session_state.use_class_weights = use_class_weights

        with ml_tabs[2]:  # Feature Engineering
            st.subheader("🔧 Advanced Feature Engineering")
            
            if 'ml_features' not in st.session_state:
                st.warning("⚠️ Please complete the setup first")
            else:
                feature_eng_tabs = st.tabs(["Interaction Features", "Advanced Selection", "Scaling & Encoding"])
                
                with feature_eng_tabs[0]:  # Interaction Features
                    st.write("**Create Interaction Features**")
                    
                    create_interactions = st.checkbox("Create interaction features")
                    
                    if create_interactions:
                        interaction_col1, interaction_col2 = st.columns(2)
                        
                        with interaction_col1:
                            max_interactions = st.slider("Maximum interactions:", 5, 20, 10)
                            interaction_method = st.radio("Method:", ["Auto (Top Correlated)", "Manual Selection"])
                        
                        with interaction_col2:
                            if interaction_method == "Manual Selection":
                                numeric_features = [f for f in st.session_state.ml_features 
                                                  if f in col_types['numeric']]
                                if len(numeric_features) >= 2:
                                    feature_pairs = []
                                    for i in range(min(5, max_interactions)):
                                        pair_col1, pair_col2 = st.columns(2)
                                        with pair_col1:
                                            f1 = st.selectbox(f"Feature 1 (pair {i+1}):", 
                                                           [""] + numeric_features, key=f"f1_{i}")
                                        with pair_col2:
                                            f2 = st.selectbox(f"Feature 2 (pair {i+1}):", 
                                                           [""] + [f for f in numeric_features if f != f1], 
                                                           key=f"f2_{i}")
                                        if f1 and f2:
                                            feature_pairs.append((f1, f2))
                        
                        if st.button("🔧 Create Interaction Features"):
                            try:
                                if interaction_method == "Auto (Top Correlated)":
                                    enhanced_df = create_interaction_features(df, max_interactions=max_interactions)
                                else:
                                    enhanced_df = create_interaction_features(df, feature_pairs=feature_pairs)
                                
                                new_features = [col for col in enhanced_df.columns 
                                              if col not in df.columns]
                                
                                if new_features:
                                    st.success(f"✅ Created {len(new_features)} interaction features")
                                    st.write("**New Features:**", ", ".join(new_features))
                                    
                                    # Update dataframe and features
                                    st.session_state.df = enhanced_df
                                    st.session_state.ml_features.extend(new_features)
                                    
                                    # Update df reference
                                    df = enhanced_df
                                    col_types = get_column_types(df)
                                else:
                                    st.info("No new interaction features created")
                            
                            except Exception as e:
                                st.error(f"Feature creation failed: {str(e)}")

                with feature_eng_tabs[1]:  # Advanced Selection  
                    st.write("**Advanced Feature Selection**")
                    
                    selection_col1, selection_col2 = st.columns(2)
                    
                    with selection_col1:
                        selection_method = st.selectbox("Selection method:", [
                            "Correlation-based",
                            "Mutual Information",
                            "Recursive Feature Elimination"
                        ])
                        
                        n_features_select = st.slider("Number of features to select:", 
                                                    1, len(st.session_state.ml_features), 
                                                    min(20, len(st.session_state.ml_features)))
                    
                    with selection_col2:
                        if st.button("🎯 Apply Feature Selection"):
                            try:
                                X_features = df[st.session_state.ml_features]
                                y_target = df[st.session_state.ml_target]
                                
                                if selection_method == "Correlation-based":
                                    selected_df = advanced_feature_selection(
                                        X_features, y_target, method='correlation', 
                                        n_features=n_features_select
                                    )
                                    selected_feature_names = selected_df.columns.tolist()
                                
                                elif selection_method == "Mutual Information":
                                    from sklearn.feature_selection import SelectKBest, mutual_info_classif
                                    selector = SelectKBest(mutual_info_classif, k=n_features_select)
                                    selector.fit(X_features.select_dtypes(include=[np.number]), y_target)
                                    selected_feature_names = X_features.select_dtypes(include=[np.number]).columns[selector.get_support()].tolist()
                                
                                elif selection_method == "Recursive Feature Elimination":
                                    from sklearn.feature_selection import RFE
                                    from sklearn.ensemble import RandomForestClassifier
                                    estimator = RandomForestClassifier(n_estimators=50, random_state=42)
                                    rfe = RFE(estimator, n_features_to_select=n_features_select)
                                    rfe.fit(X_features.select_dtypes(include=[np.number]), y_target)
                                    selected_feature_names = X_features.select_dtypes(include=[np.number]).columns[rfe.support_].tolist()
                                
                                st.success(f"✅ Selected {len(selected_feature_names)} features")
                                st.write("**Selected Features:**", ", ".join(selected_feature_names))
                                st.session_state.ml_features = selected_feature_names
                                
                            except Exception as e:
                                st.error(f"Feature selection failed: {str(e)}")

                with feature_eng_tabs[2]:  # Scaling & Encoding
                                st.write("**Preprocessing Configuration**")
                                
                                preprocess_col1, preprocess_col2 = st.columns(2)
                                
                                with preprocess_col1:
                                                scaling_method = st.selectbox("Numeric Scaling:", 
                                                                            ["StandardScaler", "MinMaxScaler", "RobustScaler", "None"])
                                                encoding_method = st.selectbox("Categorical Encoding:", 
                                                                            ["OneHotEncoder", "OrdinalEncoder", "TargetEncoder"])
                                
                                with preprocess_col2:
                                                handle_missing = st.selectbox("Missing Value Strategy:", 
                                                                            ["mean", "median", "most_frequent"])
                                                datetime_features = st.checkbox("Extract datetime features", 
                                                                                value=len(col_types['datetime']) > 0)
                                
                                # Column selection
                                numeric_features = st.multiselect("Select numeric columns for scaling:", col_types['numeric'])
                                categorical_features = st.multiselect("Select categorical columns for encoding:", col_types['categorical'])
                                
                                if st.button("✅ Apply Preprocessing"):
                                                try:
                                                                # Build numeric transformer
                                                                numeric_steps = [('imputer', SimpleImputer(strategy=handle_missing))]
                                                                if scaling_method != 'None':
                                                                                if scaling_method == 'StandardScaler':
                                                                                                numeric_steps.append(('scaler', StandardScaler()))
                                                                                elif scaling_method == 'MinMaxScaler':
                                                                                                numeric_steps.append(('scaler', MinMaxScaler()))
                                                                                elif scaling_method == 'RobustScaler':
                                                                                                numeric_steps.append(('scaler', RobustScaler()))
                                                                numeric_transformer = Pipeline(numeric_steps)
                                                                
                                                                # Build categorical transformer
                                                                categorical_steps = [('imputer', SimpleImputer(strategy='most_frequent', fill_value='Unknown'))]
                                                                if encoding_method == 'OneHotEncoder':
                                                                                categorical_steps.append(('encoder', OneHotEncoder(drop='first', sparse=False, handle_unknown='ignore')))
                                                                elif encoding_method == 'OrdinalEncoder':
                                                                                categorical_steps.append(('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)))
                                                                elif encoding_method == 'TargetEncoder':
                                                                                from category_encoders import TargetEncoder
                                                                                categorical_steps.append(('encoder', TargetEncoder()))
                                                                categorical_transformer = Pipeline(categorical_steps)
                                                                
                                                                # Combine into ColumnTransformer
                                                                preprocessor = ColumnTransformer([
                                                                                ('num', numeric_transformer, numeric_features),
                                                                                ('cat', categorical_transformer, categorical_features)
                                                                ], remainder='passthrough')
                                                                
                                                                # Save into session_state
                                                                st.session_state.preprocessor = preprocessor
                                                                st.success("✅ Preprocessing pipeline configured!")
                                                                
                                                                # ===== Preview transformed data =====
                                                                transformed_data = preprocessor.fit_transform(df)
                                                                
                                                                # Build column names for display
                                                                cat_cols_out = []
                                                                if encoding_method == 'OneHotEncoder':
                                                                                cat_cols_out = preprocessor.named_transformers_['cat']['encoder'].get_feature_names_out(categorical_features)
                                                                elif encoding_method in ['OrdinalEncoder', 'TargetEncoder']:
                                                                                cat_cols_out = categorical_features
                                                                
                                                                numeric_cols_out = numeric_features
                                                                passthrough_cols = [c for c in df.columns if c not in numeric_features + categorical_features]
                                                                all_cols = list(numeric_cols_out) + list(cat_cols_out) + list(passthrough_cols)
                                                                
                                                                transformed_df = pd.DataFrame(transformed_data, columns=all_cols)
                                                                st.write("**Preview of Transformed Data:**")
                                                                st.dataframe(transformed_df.head())
                                                                
                                                except Exception as e:
                                                                st.error(f"Error applying preprocessing: {str(e)}")

        with ml_tabs[3]:  # Model Training
            st.subheader("🏋️ Advanced Model Training")
            
            if 'ml_features' not in st.session_state:
                st.warning("⚠️ Please complete the setup first")
            else:
                # Model Selection
                st.write("**Model Selection**")
                
                if st.session_state.ml_problem_type == 'Classification':
                    available_models = [
                        "LogisticRegression", "RandomForest", "XGBoost", "LightGBM", 
                        "CatBoost", "BalancedBagging"
                    ]
                    # Filter based on availability
                    model_options = ["LogisticRegression", "RandomForest"]
                    if XGBOOST_AVAILABLE:
                        model_options.append("XGBoost")
                    if LIGHTGBM_AVAILABLE:
                        model_options.append("LightGBM")
                    if CATBOOST_AVAILABLE:
                        model_options.append("CatBoost")
                    if IMBLEARN_AVAILABLE:
                        model_options.append("BalancedBagging")
                else:
                    model_options = ["LinearRegression", "RandomForest"]
                    if XGBOOST_AVAILABLE:
                        model_options.append("XGBoost")
                    if LIGHTGBM_AVAILABLE:
                        model_options.append("LightGBM")
                    if CATBOOST_AVAILABLE:
                        model_options.append("CatBoost")
                
                selected_models = st.multiselect("Select models to train:", model_options,
                                                default=model_options[:2])
                
                # Training Options
                training_col1, training_col2, training_col3 = st.columns(3)
                
                with training_col1:
                    test_size = st.slider("Test Size (%)", 10, 40, 20) / 100
                    use_stratify = st.checkbox("Stratified Split", value=True)
                
                with training_col2:
                    tune_hyperparams = st.checkbox("Hyperparameter Tuning", 
                                                 help="Automatically tune model parameters")
                    use_class_weights = st.checkbox("Use Class Weights", 
                                                   value=st.session_state.get('use_class_weights', False))
                
                with training_col3:
                    cv_folds = st.slider("CV Folds", 3, 10, 5)
                    calibrate_probabilities = st.checkbox("Calibrate Probabilities",
                                                        help="Improve probability estimates")
                
                # Train Models Button
                if st.button("🚀 Train Advanced Models", type="primary"):
                    if selected_models:
                        with st.spinner("Training advanced models..."):
                            try:
                                # Prepare data
                                X = df[st.session_state.ml_features]
                                y = df[st.session_state.ml_target]
                                
                                # Handle resampling if configured
                                if (st.session_state.get('resampling_method') and 
                                    st.session_state.resampling_method != "None"):
                                    X_numeric = X.select_dtypes(include=[np.number])
                                    if len(X_numeric.columns) > 0:
                                        X_resampled, y_resampled = handle_imbalanced_data(
                                            X_numeric, y, st.session_state.resampling_method
                                        )
                                        # Use resampled data for training
                                        X, y = X_resampled, pd.Series(y_resampled)
                                
                                # Train-test split
                                stratify_y = y if (use_stratify and st.session_state.ml_problem_type == 'Classification') else None
                                X_train, X_test, y_train, y_test = train_test_split(
                                    X, y, test_size=test_size, random_state=42, stratify=stratify_y
                                )
                                
                                # Train models
                                results = train_advanced_models(
                                    X_train, y_train, X_test, y_test, selected_models,
                                    st.session_state.ml_problem_type, use_class_weights, tune_hyperparams
                                )
                                
                                # Store results
                                st.session_state.advanced_results = results
                                st.session_state.X_train = X_train
                                st.session_state.X_test = X_test
                                st.session_state.y_train = y_train
                                st.session_state.y_test = y_test
                                
                                st.success(f"✅ Successfully trained {len(results)} models!")
                                
                                # Show quick results
                                if results:
                                    comparison_df = create_advanced_comparison_table(results, st.session_state.ml_problem_type)
                                    st.dataframe(comparison_df, use_container_width=True)
                            
                            except Exception as e:
                                st.error(f"Training failed: {str(e)}")
                    else:
                        st.warning("Please select at least one model to train")

        with ml_tabs[4]:  # Evaluation
            st.subheader("📈 Advanced Model Evaluation")
            
            if 'advanced_results' not in st.session_state:
                st.warning("⚠️ Please train models first")
            else:
                results = st.session_state.advanced_results
                
                # Comprehensive Comparison Table
                st.write("**🏆 Comprehensive Model Comparison**")
                comparison_df = create_advanced_comparison_table(results, st.session_state.ml_problem_type)
                st.dataframe(comparison_df, use_container_width=True)
                
                # Best Model Identification
                if st.session_state.ml_problem_type == 'Classification':
                    best_model_name = comparison_df.loc[comparison_df['F1 Score'].astype(float).idxmax(), 'Model']
                    best_score = comparison_df.loc[comparison_df['F1 Score'].astype(float).idxmax(), 'F1 Score']
                    st.success(f"🥇 Best Model: **{best_model_name}** (F1 Score: {best_score})")
                else:
                    best_model_name = comparison_df.loc[comparison_df['R² Score'].astype(float).idxmax(), 'Model']
                    best_score = comparison_df.loc[comparison_df['R² Score'].astype(float).idxmax(), 'R² Score']
                    st.success(f"🥇 Best Model: **{best_model_name}** (R² Score: {best_score})")
                
                # Detailed Analysis
                analysis_model = st.selectbox("Select model for detailed analysis:", list(results.keys()))
                
                if analysis_model:
                    model_data = results[analysis_model]
                    
                    # Create detailed visualizations
                    detail_col1, detail_col2 = st.columns(2)
                    
                    with detail_col1:
                        if st.session_state.ml_problem_type == 'Classification':
                            # Enhanced Confusion Matrix
                            fig, ax = plt.subplots(figsize=(8, 6))
                            cm = confusion_matrix(st.session_state.y_test, model_data['predictions'])
                            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                            ax.set_title(f'Confusion Matrix - {analysis_model}')
                            ax.set_xlabel('Predicted')
                            ax.set_ylabel('Actual')
                            st.pyplot(fig)
                            plt.close(fig)
                            
                            # Classification Report Heatmap
                            classes = st.session_state.y_test.unique()
                            if len(classes) <= 10:  # Only for reasonable number of classes
                                report_fig = generate_classification_report_plot(
                                    st.session_state.y_test, model_data['predictions'], classes
                                )
                                st.pyplot(report_fig)
                                plt.close(report_fig)
                    
                    with detail_col2:
                        if st.session_state.ml_problem_type == 'Classification':
                            # ROC Curve
                            try:
                                if hasattr(model_data['model'], 'predict_proba'):
                                    y_proba = model_data['model'].predict_proba(st.session_state.X_test)
                                    
                                    if len(np.unique(st.session_state.y_test)) == 2:
                                        # Binary classification
                                        fpr, tpr, _ = roc_curve(st.session_state.y_test, y_proba[:, 1], pos_label = "1")
                                        auc_score = roc_auc_score(st.session_state.y_test, y_proba[:, 1])
                                        
                                        fig, ax = plt.subplots(figsize=(8, 6))
                                        ax.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.3f})')
                                        ax.plot([0, 1], [0, 1], 'r--', label='Random')
                                        ax.set_xlabel('False Positive Rate')
                                        ax.set_ylabel('True Positive Rate')
                                        ax.set_title(f'ROC Curve - {analysis_model}')
                                        ax.legend()
                                        ax.grid(True, alpha=0.3)
                                        st.pyplot(fig)
                                        plt.close(fig)
                                        
                                        # Precision-Recall Curve
                                        precision, recall, _ = precision_recall_curve(st.session_state.y_test, y_proba[:, 1])
                                        pr_auc = average_precision_score(st.session_state.y_test, y_proba[:, 1])
                                        
                                        fig, ax = plt.subplots(figsize=(8, 6))
                                        ax.plot(recall, precision, label=f'PR Curve (AUC = {pr_auc:.3f})')
                                        ax.set_xlabel('Recall')
                                        ax.set_ylabel('Precision')
                                        ax.set_title(f'Precision-Recall Curve - {analysis_model}')
                                        ax.legend()
                                        ax.grid(True, alpha=0.3)
                                        st.pyplot(fig)
                                        plt.close(fig)
                            except Exception as e:
                                st.warning(f"Could not generate probability-based plots: {str(e)}")
                        
                        else:  # Regression
                            # Predictions vs Actual
                            fig, ax = plt.subplots(figsize=(8, 6))
                            ax.scatter(st.session_state.y_test, model_data['predictions'], alpha=0.6)
                            ax.plot([st.session_state.y_test.min(), st.session_state.y_test.max()], 
                                   [st.session_state.y_test.min(), st.session_state.y_test.max()], 'r--', lw=2)
                            ax.set_xlabel('Actual Values')
                            ax.set_ylabel('Predicted Values')
                            ax.set_title(f'Predictions vs Actual - {analysis_model}')
                            st.pyplot(fig)
                            plt.close(fig)
                
                # Cross-Validation Results Visualization
                st.write("**🔄 Cross-Validation Performance**")
                cv_data = []
                for model_name, model_data in results.items():
                    for score in model_data['cv_scores']:
                        cv_data.append({'Model': model_name, 'CV_Score': score})
                
                cv_df = pd.DataFrame(cv_data)
                fig, ax = plt.subplots(figsize=(12, 6))
                sns.boxplot(data=cv_df, x='Model', y='CV_Score', ax=ax)
                ax.set_title('Cross-Validation Score Distribution')
                plt.xticks(rotation=45)
                st.pyplot(fig)
                plt.close(fig)

        with ml_tabs[5]:  # Interpretability
            st.subheader("🔍 Advanced Model Interpretability")
            
            if 'advanced_results' not in st.session_state:
                st.warning("⚠️ Please train models first")
            else:
                results = st.session_state.advanced_results
                interpret_model = st.selectbox("Select model for interpretation:", list(results.keys()))
                
                if interpret_model:
                    model_data = results[interpret_model]
                    model = model_data['model']
                    
                    interpret_tabs = st.tabs(["Feature Importance", "Permutation Importance", "Partial Dependence"])
                    
                    with interpret_tabs[0]:  # Feature Importance
                        st.write("**📊 Feature Importance Analysis**")
                        
                        # Try to get built-in feature importance
                        try:
                            if hasattr(model, 'feature_importances_'):
                                importances = model.feature_importances_
                                feature_names = st.session_state.ml_features
                                
                                importance_df = pd.DataFrame({
                                    'Feature': feature_names,
                                    'Importance': importances
                                }).sort_values('Importance', ascending=False)
                                
                                # Plot feature importance
                                fig, ax = plt.subplots(figsize=(10, 8))
                                top_features = importance_df.head(15)
                                sns.barplot(data=top_features, y='Feature', x='Importance', ax=ax)
                                ax.set_title(f'Feature Importance - {interpret_model}')
                                st.pyplot(fig)
                                plt.close(fig)
                                
                                st.dataframe(importance_df, use_container_width=True)
                                
                            elif hasattr(model, 'coef_'):
                                coef = model.coef_
                                if len(coef.shape) > 1:
                                    coef = coef[0]  # Take first class
                                
                                coef_df = pd.DataFrame({
                                    'Feature': st.session_state.ml_features,
                                    'Coefficient': coef
                                }).sort_values('Coefficient', key=abs, ascending=False)
                                
                                # Plot coefficients
                                fig, ax = plt.subplots(figsize=(10, 8))
                                top_features = coef_df.head(15)
                                colors = ['red' if x < 0 else 'blue' for x in top_features['Coefficient']]
                                sns.barplot(data=top_features, y='Feature', x='Coefficient', 
                                          palette=colors, ax=ax)
                                ax.set_title(f'Feature Coefficients - {interpret_model}')
                                ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
                                st.pyplot(fig)
                                plt.close(fig)
                                
                                st.dataframe(coef_df, use_container_width=True)
                            else:
                                st.info("Built-in feature importance not available for this model")
                        
                        except Exception as e:
                            st.error(f"Feature importance calculation failed: {str(e)}")
                    
                    with interpret_tabs[1]:  # Permutation Importance
                        st.write("**🔄 Permutation Importance (Universal Method)**")
                        
                        n_repeats = st.slider("Number of repeats:", 5, 20, 10)
                        
                        if st.button("Calculate Permutation Importance"):
                            with st.spinner("Calculating permutation importance..."):
                                try:
                                    # Use only numeric features for permutation importance
                                    X_test_numeric = st.session_state.X_test.select_dtypes(include=[np.number])
                                    
                                    if len(X_test_numeric.columns) == 0:
                                        st.error("No numeric features available for permutation importance")
                                    else:
                                        importance_df = calculate_permutation_importance(
                                            model, X_test_numeric, st.session_state.y_test, n_repeats
                                        )
                                        
                                        if importance_df is not None:
                                            # Plot permutation importance
                                            fig, ax = plt.subplots(figsize=(10, 8))
                                            top_features = importance_df.head(15)
                                            
                                            # Create error bars
                                            ax.barh(range(len(top_features)), top_features['importance'],
                                                   xerr=top_features['std'], capsize=5)
                                            ax.set_yticks(range(len(top_features)))
                                            ax.set_yticklabels(top_features['feature'])
                                            ax.set_xlabel('Permutation Importance')
                                            ax.set_title(f'Permutation Importance - {interpret_model}')
                                            ax.grid(True, alpha=0.3)
                                            st.pyplot(fig)
                                            plt.close(fig)
                                            
                                            st.dataframe(importance_df, use_container_width=True)
                                
                                except Exception as e:
                                    st.error(f"Permutation importance failed: {str(e)}")
                    
                    with interpret_tabs[2]:  # Partial Dependence
                        st.write("**📈 Partial Dependence Plots**")
                        
                        numeric_features = [f for f in st.session_state.ml_features 
                                          if f in st.session_state.X_test.select_dtypes(include=[np.number]).columns]
                        
                        if len(numeric_features) == 0:
                            st.error("No numeric features available for partial dependence plots")
                        else:
                            selected_pd_features = st.multiselect("Select features for PD plots:", 
                                                                numeric_features, 
                                                                default=numeric_features[:4])
                            
                            if st.button("Generate Partial Dependence Plots") and selected_pd_features:
                                with st.spinner("Generating partial dependence plots..."):
                                    try:
                                        X_test_numeric = st.session_state.X_test[numeric_features]
                                        pd_fig = create_partial_dependence_plots(
                                            model, X_test_numeric, selected_pd_features
                                        )
                                        
                                        if pd_fig:
                                            st.pyplot(pd_fig)
                                            plt.close(pd_fig)
                                    
                                    except Exception as e:
                                        st.error(f"Partial dependence plots failed: {str(e)}")

        with ml_tabs[6]:  # Ensemble Models
            st.subheader("🔗 Ensemble Models")
            
            if 'advanced_results' not in st.session_state:
                st.warning("⚠️ Please train individual models first")
            else:
                results = st.session_state.advanced_results
                
                st.write("**Create Ensemble Models**")
                
                ensemble_col1, ensemble_col2 = st.columns(2)
                
                with ensemble_col1:
                    ensemble_method = st.selectbox("Ensemble Method:", [
                        "Voting Classifier",
                        "Weighted Voting", 
                        "Stacking (Advanced)"
                    ])
                    
                    # Select models for ensemble
                    available_models = list(results.keys())
                    selected_ensemble_models = st.multiselect(
                        "Select models for ensemble:", 
                        available_models,
                        default=available_models[:3] if len(available_models) >= 3 else available_models
                    )
                
                with ensemble_col2:
                    if ensemble_method == "Weighted Voting":
                        st.write("**Model Weights:**")
                        weights = {}
                        for model_name in selected_ensemble_models:
                            weights[model_name] = st.slider(
                                f"{model_name} weight:", 0.1, 2.0, 1.0, 
                                key=f"weight_{model_name}"
                            )
                    
                    elif ensemble_method == "Stacking (Advanced)":
                        meta_learner = st.selectbox("Meta-learner:", [
                            "LogisticRegression", "RandomForest", "XGBoost"
                        ])
                
                if st.button("🔗 Create Ensemble Model") and len(selected_ensemble_models) >= 2:
                    with st.spinner("Creating ensemble model..."):
                        try:
                            # Prepare base models
                            base_models = {}
                            for model_name in selected_ensemble_models:
                                base_models[model_name] = results[model_name]['model']
                            
                            if ensemble_method == "Voting Classifier":
                                from sklearn.ensemble import VotingClassifier, VotingRegressor
                                estimators = [(name, model) for name, model in base_models.items()]
                                
                                if st.session_state.ml_problem_type == 'Classification':
                                    ensemble_model = VotingClassifier(estimators=estimators, voting='soft')
                                else:
                                    ensemble_model = VotingRegressor(estimators)
                            
                            elif ensemble_method == "Weighted Voting":
                                estimators = [(name, model) for name, model in base_models.items()]
                                model_weights = [weights[name] for name in selected_ensemble_models]
                                
                                if st.session_state.ml_problem_type == 'Classification':
                                    ensemble_model = VotingClassifier(
                                        estimators=estimators, voting='soft', weights=model_weights
                                    )
                                else:
                                    ensemble_model = VotingRegressor(estimators, weights=model_weights)
                            
                            elif ensemble_method == "Stacking (Advanced)":
                                from sklearn.ensemble import StackingClassifier, StackingRegressor
                                estimators = [(name, model) for name, model in base_models.items()]
                                
                                # Define meta-learner
                                if meta_learner == "LogisticRegression":
                                    if st.session_state.ml_problem_type == 'Classification':
                                        final_estimator = LogisticRegression()
                                    else:
                                        final_estimator = LinearRegression()
                                elif meta_learner == "RandomForest":
                                    if st.session_state.ml_problem_type == 'Classification':
                                        final_estimator = RandomForestClassifier(n_estimators=50)
                                    else:
                                        final_estimator = RandomForestRegressor(n_estimators=50)
                                
                                if st.session_state.ml_problem_type == 'Classification':
                                    ensemble_model = StackingClassifier(
                                        estimators=estimators, 
                                        final_estimator=final_estimator,
                                        cv=5
                                    )
                                else:
                                    ensemble_model = StackingRegressor(
                                        estimators=estimators,
                                        final_estimator=final_estimator,
                                        cv=5
                                    )
                            
                            # Train ensemble model
                            ensemble_model.fit(st.session_state.X_train, st.session_state.y_train)
                            
                            # Make predictions
                            ensemble_pred = ensemble_model.predict(st.session_state.X_test)
                            
                            # Calculate metrics
                            if st.session_state.ml_problem_type == 'Classification':
                                accuracy = accuracy_score(st.session_state.y_test, ensemble_pred)
                                precision = precision_score(st.session_state.y_test, ensemble_pred, average='weighted')
                                recall = recall_score(st.session_state.y_test, ensemble_pred, average='weighted')
                                f1 = f1_score(st.session_state.y_test, ensemble_pred, average='weighted')
                                
                                st.success(f"✅ Ensemble Model Created!")
                                st.write(f"**Accuracy:** {accuracy:.3f}")
                                st.write(f"**Precision:** {precision:.3f}")
                                st.write(f"**Recall:** {recall:.3f}")
                                st.write(f"**F1 Score:** {f1:.3f}")
                            
                            else:
                                mae = mean_absolute_error(st.session_state.y_test, ensemble_pred)
                                mse = mean_squared_error(st.session_state.y_test, ensemble_pred)
                                r2 = r2_score(st.session_state.y_test, ensemble_pred)
                                
                                st.success(f"✅ Ensemble Model Created!")
                                st.write(f"**MAE:** {mae:.3f}")
                                st.write(f"**MSE:** {mse:.3f}")
                                st.write(f"**R² Score:** {r2:.3f}")
                            
                            # Store ensemble model
                            st.session_state.ensemble_model = {
                                'model': ensemble_model,
                                'predictions': ensemble_pred,
                                'method': ensemble_method
                            }
                            
                            # Compare with individual models
                            st.write("**📊 Ensemble vs Individual Models Comparison**")
                            comparison_data = []
                            
                            # Add individual models
                            for model_name, model_data in results.items():
                                if st.session_state.ml_problem_type == 'Classification':
                                    comparison_data.append({
                                        'Model': model_name,
                                        'Type': 'Individual',
                                        'F1 Score': f"{model_data['test_f1']:.3f}",
                                        'Accuracy': f"{model_data['test_accuracy']:.3f}"
                                    })
                                else:
                                    comparison_data.append({
                                        'Model': model_name,
                                        'Type': 'Individual', 
                                        'R² Score': f"{model_data['test_r2']:.3f}",
                                        'MAE': f"{model_data['test_mae']:.3f}"
                                    })
                            
                            # Add ensemble model
                            if st.session_state.ml_problem_type == 'Classification':
                                comparison_data.append({
                                    'Model': f'Ensemble ({ensemble_method})',
                                    'Type': 'Ensemble',
                                    'F1 Score': f"{f1:.3f}",
                                    'Accuracy': f"{accuracy:.3f}"
                                })
                            else:
                                comparison_data.append({
                                    'Model': f'Ensemble ({ensemble_method})',
                                    'Type': 'Ensemble',
                                    'R² Score': f"{r2:.3f}",
                                    'MAE': f"{mae:.3f}"
                                })
                            
                            comparison_df = pd.DataFrame(comparison_data)
                            st.dataframe(comparison_df, use_container_width=True)
                        
                        except Exception as e:
                            st.error(f"Ensemble creation failed: {str(e)}")

        with ml_tabs[7]:  # Model Management
            st.subheader("💾 Advanced Model Management")
            
            if 'advanced_results' not in st.session_state:
                st.warning("⚠️ Please train models first")
            else:
                results = st.session_state.advanced_results
                
                mgmt_tabs = st.tabs(["Save Models", "Model Comparison Report", "Automated Reporting"])
                
                with mgmt_tabs[0]:  # Save Models
                    st.write("**💾 Save Trained Models**")
                    
                    save_col1, save_col2 = st.columns(2)
                    
                    with save_col1:
                        models_to_save = st.multiselect("Select models to save:", 
                                                       list(results.keys()))
                        save_format = st.selectbox("Save format:", ["joblib", "pickle"])
                        include_preprocessing = st.checkbox("Include preprocessing pipeline", value=True)
                    
                    with save_col2:
                        if st.button("💾 Prepare Models for Download"):
                            if models_to_save:
                                try:
                                    for model_name in models_to_save:
                                        model_data = results[model_name]
                                        
                                        # Create comprehensive model package
                                        model_package = {
                                            'model': model_data['model'],
                                            'model_name': model_name,
                                            'problem_type': st.session_state.ml_problem_type,
                                            'feature_names': st.session_state.ml_features,
                                            'target_name': st.session_state.ml_target,
                                            'performance_metrics': {
                                                k: v for k, v in model_data.items() 
                                                if k not in ['model', 'predictions']
                                            },
                                            'preprocessing_config': {
                                                'scaling_method': st.session_state.get('scaling_method', 'None'),
                                                'encoding_method': st.session_state.get('encoding_method', 'OneHotEncoder'),
                                                'handle_missing': st.session_state.get('handle_missing', 'mean')
                                            },
                                            'training_metadata': {
                                                'training_date': datetime.now().isoformat(),
                                                'dataset_shape': df.shape,
                                                'class_distribution': df[st.session_state.ml_target].value_counts().to_dict() if st.session_state.ml_problem_type == 'Classification' else None
                                            }
                                        }
                                        
                                        # Save to bytes
                                        buffer = BytesIO()
                                        if save_format == "joblib":
                                            joblib.dump(model_package, buffer)
                                            filename = f"{model_name.lower().replace(' ', '_')}_model.joblib"
                                        else:
                                            pickle.dump(model_package, buffer)
                                            filename = f"{model_name.lower().replace(' ', '_')}_model.pkl"
                                        
                                        buffer.seek(0)
                                        st.download_button(
                                            label=f"📥 Download {model_name}",
                                            data=buffer.getvalue(),
                                            file_name=filename,
                                            mime="application/octet-stream",
                                            key=f"download_{model_name}"
                                        )
                                    
                                    st.success(f"✅ Prepared {len(models_to_save)} models for download!")
                                
                                except Exception as e:
                                    st.error(f"Model saving failed: {str(e)}")
                
                with mgmt_tabs[1]:  # Model Comparison Report
                    st.write("**📊 Comprehensive Model Report**")
                    
                    if st.button("📊 Generate Comprehensive Report"):
                        try:
                            # Create detailed comparison
                            comparison_df = create_advanced_comparison_table(results, st.session_state.ml_problem_type)
                            
                            # Add hyperparameter information
                            for idx, model_name in enumerate(comparison_df['Model']):
                                if model_name in results:
                                    best_params = results[model_name].get('best_params', {})
                                    comparison_df.at[idx, 'Best_Params'] = str(best_params) if best_params else 'Default'
                            
                            # Save comprehensive report
                            buffer = BytesIO()
                            comparison_df.to_csv(buffer, index=False)
                            buffer.seek(0)
                            
                            st.download_button(
                                "📥 Download Model Comparison Report",
                                buffer.getvalue(),
                                file_name=f"ml_model_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )
                            
                            # Display summary statistics
                            st.write("**📈 Model Performance Summary**")
                            st.dataframe(comparison_df, use_container_width=True)
                            
                        except Exception as e:
                            st.error(f"Report generation failed: {str(e)}")
                
                with mgmt_tabs[2]:  # Automated Reporting
                    st.write("**📋 Automated ML Report**")
                    
                    if st.button("🤖 Generate Automated Analysis Report"):
                        try:
                            # Create comprehensive analysis report
                            report_sections = []
                            
                            # Dataset Summary
                            report_sections.append("=== DATASET ANALYSIS REPORT ===\n")
                            report_sections.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                            
                            report_sections.append("1. DATASET OVERVIEW\n")
                            report_sections.append(f"   - Shape: {df.shape[0]:,} rows × {df.shape[1]} columns\n")
                            report_sections.append(f"   - Target Variable: {st.session_state.ml_target}\n")
                            report_sections.append(f"   - Problem Type: {st.session_state.ml_problem_type}\n")
                            report_sections.append(f"   - Selected Features: {len(st.session_state.ml_features)}\n\n")
                            
                            # Data Quality Assessment
                            report_sections.append("2. DATA QUALITY ASSESSMENT\n")
                            missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
                            duplicates = df.duplicated().sum()
                            report_sections.append(f"   - Missing Data: {missing_pct:.1f}%\n")
                            report_sections.append(f"   - Duplicate Records: {duplicates:,}\n")
                            
                            if st.session_state.ml_problem_type == 'Classification':
                                is_imbalanced, imbalance_info = check_class_imbalance(df[st.session_state.ml_target])
                                if is_imbalanced:
                                    report_sections.append(f"   - Class Imbalance: {imbalance_info['ratio']:.2f}:1 ratio detected\n")
                                else:
                                    report_sections.append("   - Class Balance: Relatively balanced\n")
                            
                            report_sections.append("\n")
                            
                            # Model Performance Summary
                            report_sections.append("3. MODEL PERFORMANCE SUMMARY\n")
                            comparison_df = create_advanced_comparison_table(results, st.session_state.ml_problem_type)
                            
                            if st.session_state.ml_problem_type == 'Classification':
                                best_model = comparison_df.loc[comparison_df['F1 Score'].astype(float).idxmax(), 'Model']
                                best_f1 = comparison_df.loc[comparison_df['F1 Score'].astype(float).idxmax(), 'F1 Score']
                                report_sections.append(f"   - Best Performing Model: {best_model} (F1: {best_f1})\n")
                                
                                for _, row in comparison_df.iterrows():
                                    report_sections.append(f"   - {row['Model']}: F1={row['F1 Score']}, Accuracy={row['Accuracy']}, Precision={row['Precision']}, Recall={row['Recall']}\n")
                            else:
                                best_model = comparison_df.loc[comparison_df['R² Score'].astype(float).idxmax(), 'Model']
                                best_r2 = comparison_df.loc[comparison_df['R² Score'].astype(float).idxmax(), 'R² Score']
                                report_sections.append(f"   - Best Performing Model: {best_model} (R²: {best_r2})\n")
                                
                                for _, row in comparison_df.iterrows():
                                    report_sections.append(f"   - {row['Model']}: R²={row['R² Score']}, MAE={row['MAE']}, RMSE={row['RMSE']}\n")
                            
                            report_sections.append("\n")
                            
                            # Recommendations
                            report_sections.append("4. RECOMMENDATIONS\n")
                            
                            # Data recommendations
                            if missing_pct > 10:
                                report_sections.append("   - Consider advanced imputation techniques for missing data\n")
                            if duplicates > 0:
                                report_sections.append("   - Remove duplicate records to improve data quality\n")
                            
                            # Model recommendations
                            if st.session_state.ml_problem_type == 'Classification':
                                f1_scores = comparison_df['F1 Score'].astype(float)
                                if f1_scores.max() < 0.7:
                                    report_sections.append("   - Consider feature engineering or ensemble methods to improve performance\n")
                                if st.session_state.get('ml_is_imbalanced', False):
                                    report_sections.append("   - Apply class balancing techniques for better minority class prediction\n")
                            else:
                                r2_scores = comparison_df['R² Score'].astype(float)
                                if r2_scores.max() < 0.8:
                                    report_sections.append("   - Consider polynomial features or regularization techniques\n")
                            
                            report_sections.append("   - Use cross-validation for more robust model evaluation\n")
                            report_sections.append("   - Consider hyperparameter tuning for optimal performance\n")
                            report_sections.append("   - Implement ensemble methods for improved predictions\n\n")
                            
                            # Technical Details
                            report_sections.append("5. TECHNICAL CONFIGURATION\n")
                            report_sections.append(f"   - Scaling Method: {st.session_state.get('scaling_method', 'None')}\n")
                            report_sections.append(f"   - Encoding Method: {st.session_state.get('encoding_method', 'OneHotEncoder')}\n")
                            report_sections.append(f"   - Missing Value Strategy: {st.session_state.get('handle_missing', 'mean')}\n")
                            if st.session_state.get('use_class_weights', False):
                                report_sections.append("   - Class Weights: Enabled\n")
                            report_sections.append("\n")
                            
                            # Combine all sections
                            full_report = "".join(report_sections)
                            
                            # Create downloadable report
                            report_buffer = BytesIO()
                            report_buffer.write(full_report.encode('utf-8'))
                            report_buffer.seek(0)
                            
                            st.download_button(
                                "📥 Download Automated Analysis Report",
                                report_buffer.getvalue(),
                                file_name=f"ml_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                                mime="text/plain"
                            )
                            
                            # Display report preview
                            st.write("**📄 Report Preview:**")
                            st.text_area("Generated Report:", full_report, height=400)
                            
                        except Exception as e:
                            st.error(f"Automated report generation failed: {str(e)}")

    # ================================
    # Advanced Predictions Section
    # ================================
    st.header("9. 🔮 Advanced Predictions & Deployment")
    
    if 'advanced_results' in st.session_state:
        pred_tabs = st.tabs(["Single Prediction", "Batch Predictions", "Model Deployment Info"])
        
        with pred_tabs[0]:  # Single Prediction
            st.subheader("🎯 Interactive Single Prediction")
            
            results = st.session_state.advanced_results
            prediction_model = st.selectbox("Select model for prediction:", list(results.keys()))
            
            if prediction_model:
                st.write("**Enter feature values for prediction:**")
                
                # Create input fields
                input_values = {}
                pred_cols = st.columns(3)
                
                for i, feature in enumerate(st.session_state.ml_features):
                    col = pred_cols[i % 3]
                    
                    with col:
                        if feature in col_types['numeric']:
                            # Use statistics for better default values
                            feature_stats = df[feature].describe()
                            input_values[feature] = st.number_input(
                                f"{feature}:",
                                value=float(feature_stats['50%']),  # median as default
                                help=f"Range: {feature_stats['min']:.2f} - {feature_stats['max']:.2f}"
                            )
                        elif feature in col_types['categorical']:
                            unique_vals = df[feature].dropna().unique()
                            input_values[feature] = st.selectbox(f"{feature}:", unique_vals)
                        elif feature in col_types['boolean']:
                            input_values[feature] = st.checkbox(f"{feature}:")
                        else:
                            input_values[feature] = st.text_input(f"{feature}:")
                
                pred_col1, pred_col2 = st.columns(2)
                
                with pred_col1:
                    if st.button("🔮 Make Prediction", type="primary"):
                        try:
                            # Create prediction dataframe
                            pred_df = pd.DataFrame([input_values])
                            
                            # Make prediction
                            model = results[prediction_model]['model']
                            prediction = model.predict(pred_df)[0]
                            
                            st.success(f"**Prediction: {prediction}**")
                            
                            # Show probability for classification
                            if st.session_state.ml_problem_type == 'Classification':
                                try:
                                    if hasattr(model, 'predict_proba'):
                                        probabilities = model.predict_proba(pred_df)[0]
                                        
                                        st.write("**Class Probabilities:**")
                                        prob_data = []
                                        for i, prob in enumerate(probabilities):
                                            class_label = model.classes_[i]
                                            prob_data.append({
                                                'Class': class_label,
                                                'Probability': f"{prob:.3f}",
                                                'Percentage': f"{prob*100:.1f}%"
                                            })
                                        
                                        prob_df = pd.DataFrame(prob_data)
                                        st.dataframe(prob_df, use_container_width=True)
                                        
                                        # Probability visualization
                                        fig, ax = plt.subplots(figsize=(8, 5))
                                        ax.barh(range(len(probabilities)), probabilities)
                                        ax.set_yticks(range(len(probabilities)))
                                        ax.set_yticklabels(model.classes_)
                                        ax.set_xlabel('Probability')
                                        ax.set_title('Prediction Probabilities')
                                        st.pyplot(fig)
                                        plt.close(fig)
                                
                                except Exception as e:
                                    st.warning(f"Could not generate probabilities: {str(e)}")
                        
                        except Exception as e:
                            st.error(f"Prediction failed: {str(e)}")
                
                with pred_col2:
                    # Prediction confidence and explanation
                    st.write("**Prediction Insights:**")
                    if st.button("🔍 Explain Prediction"):
                        try:
                            # Simple feature contribution explanation
                            pred_df = pd.DataFrame([input_values])
                            
                            # For tree-based models, try to get feature importance
                            model = results[prediction_model]['model']
                            if hasattr(model, 'feature_importances_'):
                                feature_importance = model.feature_importances_
                                
                                # Create feature contribution visualization
                                contrib_data = []
                                for i, feature in enumerate(st.session_state.ml_features):
                                    if i < len(feature_importance):
                                        contrib_data.append({
                                            'Feature': feature,
                                            'Value': input_values.get(feature, 'N/A'),
                                            'Importance': f"{feature_importance[i]:.3f}"
                                        })
                                
                                contrib_df = pd.DataFrame(contrib_data).sort_values('Importance', ascending=False)
                                st.write("**Feature Contributions:**")
                                st.dataframe(contrib_df.head(10), use_container_width=True)
                        
                        except Exception as e:
                            st.warning(f"Prediction explanation failed: {str(e)}")

        with pred_tabs[1]:  # Batch Predictions
            st.subheader("📁 Advanced Batch Predictions")
            
            results = st.session_state.advanced_results
            batch_model = st.selectbox("Select model for batch predictions:", 
                                     list(results.keys()), key="batch_model")
            
            batch_file = st.file_uploader("Upload CSV for batch predictions:", type=['csv'])
            
            if batch_file and batch_model:
                try:
                    batch_df = pd.read_csv(batch_file)
                    st.write(f"**Uploaded data shape:** {batch_df.shape}")
                    st.dataframe(batch_df.head(), use_container_width=True)
                    
                    # Check feature compatibility
                    missing_features = set(st.session_state.ml_features) - set(batch_df.columns)
                    extra_features = set(batch_df.columns) - set(st.session_state.ml_features)
                    
                    if missing_features:
                        st.error(f"❌ Missing features: {', '.join(missing_features)}")
                    else:
                        if extra_features:
                            st.info(f"ℹ️ Extra features will be ignored: {', '.join(list(extra_features)[:5])}")
                        
                        batch_col1, batch_col2 = st.columns(2)
                        
                        with batch_col1:
                            include_probabilities = st.checkbox("Include prediction probabilities", 
                                                              value=st.session_state.ml_problem_type == 'Classification')
                            add_confidence_scores = st.checkbox("Add confidence indicators")
                        
                        with batch_col2:
                            prediction_threshold = 0.5
                            if st.session_state.ml_problem_type == 'Classification':
                                prediction_threshold = st.slider("Prediction threshold:", 0.1, 0.9, 0.5)
                        
                        if st.button("🔮 Generate Batch Predictions", type="primary"):
                            try:
                                with st.spinner("Generating predictions..."):
                                    # Select and prepare features
                                    X_batch = batch_df[st.session_state.ml_features]
                                    
                                    # Make predictions
                                    model = results[batch_model]['model']
                                    predictions = model.predict(X_batch)
                                    
                                    # Create results dataframe
                                    results_df = batch_df.copy()
                                    results_df['Predicted'] = predictions
                                    
                                    # Add probabilities for classification
                                    if (st.session_state.ml_problem_type == 'Classification' and 
                                        include_probabilities and hasattr(model, 'predict_proba')):
                                        
                                        probabilities = model.predict_proba(X_batch)
                                        
                                        # Add probability columns
                                        for i, class_label in enumerate(model.classes_):
                                            results_df[f'Prob_{class_label}'] = probabilities[:, i]
                                        
                                        # Add max probability as confidence
                                        if add_confidence_scores:
                                            results_df['Confidence'] = probabilities.max(axis=1)
                                            results_df['Confidence_Level'] = pd.cut(
                                                results_df['Confidence'],
                                                bins=[0, 0.6, 0.8, 1.0],
                                                labels=['Low', 'Medium', 'High']
                                            )
                                    
                                    st.success(f"✅ Generated {len(predictions)} predictions!")
                                    
                                    # Display results summary
                                    summary_col1, summary_col2 = st.columns(2)
                                    
                                    with summary_col1:
                                        st.write("**Prediction Summary:**")
                                        if st.session_state.ml_problem_type == 'Classification':
                                            pred_counts = pd.Series(predictions).value_counts()
                                            for class_name, count in pred_counts.items():
                                                pct = (count / len(predictions)) * 100
                                                st.write(f"• {class_name}: {count} ({pct:.1f}%)")
                                        else:
                                            pred_stats = pd.Series(predictions).describe()
                                            st.write(f"• Mean: {pred_stats['mean']:.3f}")
                                            st.write(f"• Std: {pred_stats['std']:.3f}")
                                            st.write(f"• Range: {pred_stats['min']:.3f} - {pred_stats['max']:.3f}")
                                    
                                    with summary_col2:
                                        if add_confidence_scores and 'Confidence_Level' in results_df.columns:
                                            st.write("**Confidence Distribution:**")
                                            conf_counts = results_df['Confidence_Level'].value_counts()
                                            for level, count in conf_counts.items():
                                                pct = (count / len(results_df)) * 100
                                                st.write(f"• {level}: {count} ({pct:.1f}%)")
                                    
                                    # Display sample results
                                    st.write("**Sample Predictions:**")
                                    st.dataframe(results_df.head(10), use_container_width=True)
                                    
                                    # Download results
                                    csv_buffer = BytesIO()
                                    results_df.to_csv(csv_buffer, index=False)
                                    csv_buffer.seek(0)
                                    
                                    st.download_button(
                                        "📥 Download Batch Predictions",
                                        csv_buffer.getvalue(),
                                        file_name=f"batch_predictions_{batch_model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                        mime="text/csv"
                                    )
                            
                            except Exception as e:
                                st.error(f"Batch prediction failed: {str(e)}")
                
                except Exception as e:
                    st.error(f"Error reading batch file: {str(e)}")

        with pred_tabs[2]:  # Model Deployment Info
            st.subheader("🚀 Model Deployment Information")
            
            st.write("**Deployment Readiness Checklist:**")
            
            deployment_col1, deployment_col2 = st.columns(2)
            
            with deployment_col1:
                st.write("**✅ Model Requirements:**")
                st.write("• Models trained and validated")
                st.write("• Performance metrics documented")
                st.write("• Feature preprocessing pipeline defined")
                st.write("• Model artifacts saved")
                
                if 'advanced_results' in st.session_state:
                    st.success("All requirements met!")
                else:
                    st.warning("Train models first")
            
            with deployment_col2:
                st.write("**📋 Deployment Checklist:**")
                
                # Model selection for deployment
                if 'advanced_results' in st.session_state:
                    results = st.session_state.advanced_results
                    deployment_model = st.selectbox("Select model for deployment:", list(results.keys()))
                    
                    if deployment_model:
                        model_data = results[deployment_model]
                        
                        st.write("**Model Specifications:**")
                        st.write(f"• Model Type: {deployment_model}")
                        st.write(f"• Problem Type: {st.session_state.ml_problem_type}")
                        st.write(f"• Features: {len(st.session_state.ml_features)}")
                        
                        if st.session_state.ml_problem_type == 'Classification':
                            st.write(f"• F1 Score: {model_data['test_f1']:.3f}")
                            st.write(f"• Accuracy: {model_data['test_accuracy']:.3f}")
                        else:
                            st.write(f"• R² Score: {model_data['test_r2']:.3f}")
                            st.write(f"• MAE: {model_data['test_mae']:.3f}")
            
            # Deployment code example
            st.write("**💻 Sample Deployment Code:**")
            
            if 'advanced_results' in st.session_state:
                deployment_code = f'''
# Sample deployment code for {deployment_model if 'deployment_model' in locals() else 'your model'}
import joblib
import pandas as pd
import numpy as np

# Load the saved model
model_package = joblib.load('path_to_your_model.joblib')
model = model_package['model']
feature_names = model_package['feature_names']

# Example prediction function
def make_prediction(input_data):
    """
    Make prediction using the trained model
    
    Args:
        input_data (dict): Dictionary containing feature values
    
    Returns:
        prediction: Model prediction
        probability: Prediction probability (for classification)
    """
    # Create DataFrame with correct feature order
    input_df = pd.DataFrame([input_data])[feature_names]
    
    # Make prediction
    prediction = model.predict(input_df)[0]
    
    # Get probability for classification
    probability = None
    if hasattr(model, 'predict_proba'):
        probability = model.predict_proba(input_df)[0]
    
    return prediction, probability

# Example usage
sample_input = {{
    {', '.join([f'"{feature}": 0.0' for feature in st.session_state.ml_features[:5]])}
    # ... add all required features
}}

prediction, probability = make_prediction(sample_input)
print(f"Prediction: {{prediction}}")
if probability is not None:
    print(f"Probabilities: {{probability}}")
'''
                
                st.code(deployment_code, language='python')
                
                # Download deployment template
                if st.button("📥 Download Deployment Template"):
                    template_buffer = BytesIO()
                    template_buffer.write(deployment_code.encode('utf-8'))
                    template_buffer.seek(0)
                    
                    st.download_button(
                        "Download Python Template",
                        template_buffer.getvalue(),
                        file_name=f"model_deployment_template.py",
                        mime="text/plain"
                    )

    # ================================
    # AI Assistant Integration
    # ================================
    st.header("10. 🤖 AI Data Assistant (Coming Soon)")
    
    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Predefined analysis suggestions
    st.subheader("💡 Quick Analysis Suggestions")
    suggestions = [
        "What are the main patterns in this dataset?",
        "Which columns have the most missing values and what should I do about them?",
        "Are there any correlations I should be aware of?",
        "What data quality issues should I address first?",
        "Can you suggest the best visualization for my data?",
        "What preprocessing steps do you recommend?"
    ]
    
    suggestion_cols = st.columns(2)
    for i, suggestion in enumerate(suggestions):
        col = suggestion_cols[i % 2]
        if col.button(f"💭 {suggestion}", key=f"suggestion_{i}"):
            st.session_state.selected_query = suggestion

    # Chat interface
    st.subheader("💬 Chat with AI Assistant")
    
    # Display dataset summary for context
    dataset_context = {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'numeric_columns': col_types['numeric'],
        'categorical_columns': col_types['categorical'],
        'sample_data': df.head(3).to_dict()
    }
    
    # User input
    user_query = st.text_area(
        "Ask me anything about your dataset:",
        value=st.session_state.get('selected_query', ''),
        height=100,
        placeholder="e.g., 'What preprocessing steps do you recommend for this dataset?'"
    )
    
    # Clear selected query after use
    if 'selected_query' in st.session_state:
        del st.session_state.selected_query
    
    col1, col2 = st.columns([1, 4])
    with col1:
        send_query = st.button("🚀 Send Query", type="primary")
    with col2:
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()

    if send_query and user_query:
        # Generate AI response
        response = generate_ai_response(user_query, dataset_context, df)
        
        # Add to chat history
        st.session_state.chat_history.append({
            'query': user_query,
            'response': response
        })

    # Display chat history
    if st.session_state.chat_history:
        st.subheader("💬 Conversation History")
        for i, chat in enumerate(reversed(st.session_state.chat_history[-5:])):  # Show last 5
            with st.expander(f"Query {len(st.session_state.chat_history)-i}: {chat['query'][:60]}..."):
                st.write("**Your Question:**")
                st.write(chat['query'])
                st.write("**AI Response:**")
                st.write(chat['response'])

    # ================================
    # Data Export
    # ================================
    st.header("10. 💾 Export Cleaned Dataset")
    
    export_col1, export_col2 = st.columns(2)
    
    with export_col1:
        export_format = st.selectbox("Select export format", ["CSV", "Excel", "JSON", "Parquet"])
        
    with export_col2:
        filename = st.text_input("Filename (without extension)", value="cleaned_dataset")
    
    if st.button("📤 Export Dataset", type="primary"):
        try:
            if export_format == "CSV":
                csv_data = df.to_csv(index=False)
                st.download_button(
                    "Download CSV",
                    csv_data,
                    file_name=f"{filename}.csv",
                    mime="text/csv"
                )
            elif export_format == "Excel":
                buffer = BytesIO()
                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                    df.to_excel(writer, sheet_name='Cleaned_Data', index=False)
                st.download_button(
                    "Download Excel",
                    buffer.getvalue(),
                    file_name=f"{filename}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            elif export_format == "JSON":
                json_data = df.to_json(indent=2, orient='records')
                st.download_button(
                    "Download JSON",
                    json_data,
                    file_name=f"{filename}.json",
                    mime="application/json"
                )
            elif export_format == "Parquet":
                buffer = BytesIO()
                df.to_parquet(buffer, index=False)
                st.download_button(
                    "Download Parquet",
                    buffer.getvalue(),
                    file_name=f"{filename}.parquet",
                    mime="application/octet-stream"
                )
            
            st.success(f"✅ Dataset prepared for download in {export_format} format!")
            
        except Exception as e:
            st.error(f"❌ Error exporting dataset: {str(e)}")

    # ================================
    # Data Processing Summary
    # ================================
    st.header("11. 📋 Processing Summary")
    
    if "original_df" in st.session_state:
        original_shape = st.session_state.original_df.shape
        current_shape = df.shape
        
        summary_col1, summary_col2 = st.columns(2)
        
        with summary_col1:
            st.write("**Original Dataset:**")
            st.write(f"• Rows: {original_shape[0]:,}")
            st.write(f"• Columns: {original_shape[1]:,}")
            st.write(f"• Memory: {st.session_state.original_df.memory_usage(deep=True).sum()/(1024**2):.2f} MB")
            
        with summary_col2:
            st.write("**Current Dataset:**")
            st.write(f"• Rows: {current_shape[0]:,}")
            st.write(f"• Columns: {current_shape[1]:,}")
            st.write(f"• Memory: {df.memory_usage(deep=True).sum()/(1024**2):.2f} MB")
        
        # Changes summary
        row_change = current_shape[0] - original_shape[0]
        col_change = current_shape[1] - original_shape[1]
        
        st.write("**Changes Made:**")
        if row_change != 0:
            change_type = "added" if row_change > 0 else "removed"
            st.write(f"• Rows {change_type}: {abs(row_change):,}")
        if col_change != 0:
            change_type = "added" if col_change > 0 else "removed"
            st.write(f"• Columns {change_type}: {abs(col_change):,}")
        
        if row_change == 0 and col_change == 0:
            st.info("No structural changes made to the dataset")

if uploaded_file is None:
    st.info("👆 Please upload a dataset to get started with data analysis and cleaning!")

# ================================
# Footer
# ================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🛠️ <strong>Enhanced Data Analysis Tool</strong> | Built with ❤️ using Streamlit, Pandas, Seaborn & Matplotlib</p>
    <p>💡 <em>Tip: Use the AI assistant for personalized insights about your dataset!</em></p>
</div>
""", unsafe_allow_html=True)
