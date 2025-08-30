"""
Simplified IoT Ransomware Detection Pipeline
Following the exact requirements specified
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import joblib

def main():
    # 1. Load the dataset into pandas
    print("1. Loading dataset...")
    df = pd.read_csv(r"c:\Users\MADHURIMA\Desktop\ml\dataset\CTU-IoT-ramsomware -Capture-1-1conn.log.labeled.csv")
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # 2. Drop useless columns
    print("\n2. Dropping useless columns...")
    columns_to_drop = ['Unnamed: 0', 'id.orig_h', 'id.resp_h']
    existing_cols_to_drop = [col for col in columns_to_drop if col in df.columns]
    df = df.drop(columns=existing_cols_to_drop)
    print(f"Dropped columns: {existing_cols_to_drop}")
    print(f"Remaining columns: {list(df.columns)}")
    
    # 3. Handle missing values
    print("\n3. Handling missing values...")
    print("Missing values before:")
    print(df.isnull().sum())
    
    # Fill numeric columns with 0
    numeric_columns = ['duration', 'orig_bytes', 'resp_bytes']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(0)
    
    # Fill other numeric columns
    other_numeric = ['missed_bytes', 'orig_pkts', 'orig_ip_bytes', 'resp_pkts', 'resp_ip_bytes']
    for col in other_numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(0)
    
    print("Missing values after:")
    print(df.isnull().sum())
    
    # 4. Encode categorical columns using one-hot encoding
    print("\n4. Encoding categorical columns...")
    
    # Separate features and target
    X = df.drop('label', axis=1)
    y = df['label']
    
    # Encode target variable (Benign=0, Malicious=1)
    y_encoded = (y == 'Malicious').astype(int)
    
    # One-hot encode categorical variables
    categorical_columns = ['proto', 'service', 'conn_state', 'history']
    existing_categorical = [col for col in categorical_columns if col in X.columns]
    
    # Fill missing categorical values
    for col in existing_categorical:
        X[col] = X[col].fillna('unknown')
    
    X_encoded = pd.get_dummies(X, columns=existing_categorical, prefix=existing_categorical)
    
    print(f"Features after encoding: {X_encoded.shape[1]}")
    print(f"Target distribution: {pd.Series(y_encoded).value_counts()}")
    
    # 5. Split data into train/test (80/20, stratified on label)
    print("\n5. Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # 6. Scale numeric features using StandardScaler
    print("\n6. Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrame for feature names
    feature_names = X_encoded.columns.tolist()
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_names)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_names)
    
    # 7. Balance classes using SMOTE oversampling
    print("\n7. Balancing classes with SMOTE...")
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
    
    print("Class distribution before SMOTE:")
    unique, counts = np.unique(y_train, return_counts=True)
    for class_val, count in zip(unique, counts):
        print(f"Class {class_val}: {count} samples")
    
    print("Class distribution after SMOTE:")
    unique, counts = np.unique(y_train_balanced, return_counts=True)
    for class_val, count in zip(unique, counts):
        print(f"Class {class_val}: {count} samples")
    
    # 8. Train XGBoost model (usually gives better results)
    print("\n8. Training XGBoost model...")
    
    # Basic XGBoost model
    xgb_model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1
    )
    
    xgb_model.fit(X_train_balanced, y_train_balanced)
    print("XGBoost model trained successfully!")
    
    # Also train Random Forest for comparison
    print("\nTraining Random Forest model...")
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    
    rf_model.fit(X_train_scaled, y_train)  # Use original training set with class weights
    print("Random Forest model trained successfully!")
    
    # 9. Optimize hyperparameters (simplified version)
    print("\n9. Hyperparameter optimization for XGBoost...")
    
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 6],
        'learning_rate': [0.1, 0.2]
    }
    
    grid_search = GridSearchCV(
        xgb.XGBClassifier(random_state=42, n_jobs=-1),
        param_grid,
        cv=3,
        scoring='f1',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train_balanced, y_train_balanced)
    best_xgb = grid_search.best_estimator_
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation F1 score: {grid_search.best_score_:.4f}")
    
    # 10. Evaluate using Precision, Recall, F1-score, and ROC-AUC
    print("\n10. Model Evaluation...")
    
    def evaluate_model(model, X_test, y_test, model_name):
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        print(f"\n{model_name} Results:")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"ROC-AUC: {roc_auc:.4f}")
        
        print(f"\nDetailed Classification Report for {model_name}:")
        print(classification_report(y_test, y_pred, target_names=['Benign', 'Malicious']))
        
        return precision, recall, f1, roc_auc
    
    # Evaluate XGBoost
    xgb_precision, xgb_recall, xgb_f1, xgb_roc_auc = evaluate_model(
        xgb_model, X_test_scaled, y_test, "XGBoost (Basic)"
    )
    
    # Evaluate optimized XGBoost
    best_xgb_precision, best_xgb_recall, best_xgb_f1, best_xgb_roc_auc = evaluate_model(
        best_xgb, X_test_scaled, y_test, "XGBoost (Optimized)"
    )
    
    # Evaluate Random Forest
    rf_precision, rf_recall, rf_f1, rf_roc_auc = evaluate_model(
        rf_model, X_test_scaled, y_test, "Random Forest"
    )
    
    # 11. Save the best model with joblib
    print("\n11. Saving models...")
    
    # Determine best model
    models_performance = {
        'XGBoost_Basic': xgb_f1,
        'XGBoost_Optimized': best_xgb_f1,
        'Random_Forest': rf_f1
    }
    
    best_model_name = max(models_performance, key=models_performance.get)
    print(f"Best model based on F1-score: {best_model_name}")
    
    if best_model_name == 'XGBoost_Optimized':
        best_model = best_xgb
    elif best_model_name == 'XGBoost_Basic':
        best_model = xgb_model
    else:
        best_model = rf_model
    
    # Save the best model, scaler, and feature names
    joblib.dump(best_model, 'best_iot_ransomware_model.joblib')
    joblib.dump(scaler, 'iot_ransomware_scaler.joblib')
    joblib.dump(feature_names, 'iot_ransomware_features.joblib')
    
    print("Models saved successfully!")
    print("Files created:")
    print("- best_iot_ransomware_model.joblib")
    print("- iot_ransomware_scaler.joblib")
    print("- iot_ransomware_features.joblib")
    
    # Summary
    print("\n" + "="*60)
    print("FINAL RESULTS SUMMARY")
    print("="*60)
    print(f"XGBoost (Basic)    - F1: {xgb_f1:.4f}, ROC-AUC: {xgb_roc_auc:.4f}")
    print(f"XGBoost (Optimized) - F1: {best_xgb_f1:.4f}, ROC-AUC: {best_xgb_roc_auc:.4f}")
    print(f"Random Forest      - F1: {rf_f1:.4f}, ROC-AUC: {rf_roc_auc:.4f}")
    print(f"\nBest Model: {best_model_name}")
    print("="*60)

if __name__ == "__main__":
    main()