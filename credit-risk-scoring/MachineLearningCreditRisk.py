'''
Project: Machine Learning Credit Risk Modeling (Credit Scoring)
Description: An end-to-end Machine Learning pipeline to predict loan defaults.
Processes a standard Kaggle credit risk dataset, applies feature engineering, 
trains a Random Forest Classifier, and visualizes feature importance.
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# 1. Data Ingestion & Preprocessing
def load_and_preprocess_data(file_path):
    """
    Loads a Kaggle credit risk dataset, handles missing values, encodes 
    categorical variables, and scales numerical features.
    """
    print(f"Loading dataset from: {file_path}")
    # Expected target column: 'loan_status' (1 = Default, 0 = Paid)
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print("Error: Kaggle dataset not found. Please verify the file path.")
        return None, None, None, None, None

    # Drop uninformative identification columns if they exist
    cols_to_drop = ['id', 'member_id']
    df.drop(columns=[col for col in cols_to_drop if col in df.columns], inplace=True)

    # Separate features and target
    target_col = 'loan_status' 
    if target_col not in df.columns:
        # Fallback to 'Default' if using a differently structured Kaggle dataset
        target_col = 'Default' 
    
    y = df[target_col]
    X = df.drop(columns=[target_col])

    # Handle Categorical Variables (One-Hot Encoding)
    X = pd.get_dummies(X, drop_first=True)

    # Train-Test Split (Perform before imputation/scaling to prevent data leakage)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    # Handle Missing Values (Imputation)
    imputer = SimpleImputer(strategy='median')
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)

    # Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)

    feature_names = X.columns.tolist()

    return X_train_scaled, X_test_scaled, y_train, y_test, feature_names

# 2. Model Training
def train_credit_model(X_train, y_train):
    """
    Trains a Random Forest ensemble model to predict credit defaults,
    utilizing class weights to handle imbalanced financial datasets.
    """
    print("Training Random Forest Classifier...")
    model = RandomForestClassifier(
        n_estimators=100, 
        max_depth=10, 
        random_state=42, 
        class_weight='balanced',
        n_jobs=-1 # Utilize all CPU cores
    )
    model.fit(X_train, y_train)
    return model

# 3. Evaluation & Metrics
def evaluate_model(model, X_test, y_test):
    """
    Evaluates the model's predictive power using accuracy, recall, and ROC-AUC.
    """
    predictions = model.predict(X_test)
    prob_predictions = model.predict_proba(X_test)[:, 1]
    
    print("\n" + "=" * 50)
    print("MODEL EVALUATION REPORT")
    print("=" * 50)
    print(classification_report(y_test, predictions))
    
    auc_score = roc_auc_score(y_test, prob_predictions)
    print(f"ROC-AUC Score: {auc_score:.4f}\n")
    
    # Plot ROC Curve
    fpr, tpr, _ = roc_curve(y_test, prob_predictions)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {auc_score:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) - Credit Risk')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.show()

# 4. Feature Importance Visualization
def plot_feature_importance(model, feature_names, top_n=15):
    """
    Extracts and visualizes the top N most critical financial variables 
    driving the model's decisions for explainability.
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    sorted_features = [feature_names[i] for i in indices]
    
    plt.figure(figsize=(10, 8))
    sns.barplot(x=importances[indices], y=sorted_features, palette='viridis')
    plt.title(f'Top {top_n} Feature Importances: Drivers of Default Risk')
    plt.xlabel('Relative Importance Score')
    plt.ylabel('Financial Feature')
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.show()

# 5. Main Execution Block
if __name__ == '__main__':
    
    KAGGLE_FILE_PATH = 'kaggle_credit_risk_dataset.csv' 
    
    X_train, X_test, y_train, y_test, features = load_and_preprocess_data(KAGGLE_FILE_PATH)
    
    if X_train is not None:
        risk_model = train_credit_model(X_train, y_train)
        evaluate_model(risk_model, X_test, y_test)
        plot_feature_importance(risk_model, features)
