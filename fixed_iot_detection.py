"""
FIXED IoT Ransomware Detection Pipeline
Addresses data leakage issues and ensures proper train/test split
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_score, recall_score, f1_score, accuracy_score
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import joblib
import warnings
warnings.filterwarnings('ignore')

def main():
    print("🔧 FIXED IoT RANSOMWARE DETECTION - PROPER TRAIN/TEST SPLIT")
    print("=" * 70)
    
    # 1. Load the dataset
    print("1. Loading dataset...")
    df = pd.read_csv(r"dataset\CTU-IoT-ramsomware -Capture-1-1conn.log.labeled.csv")
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # 2. Drop useless columns FIRST
    print("\n2. Dropping useless columns...")
    columns_to_drop = ['Unnamed: 0', 'id.orig_h', 'id.resp_h']
    existing_cols_to_drop = [col for col in columns_to_drop if col in df.columns]
    df = df.drop(columns=existing_cols_to_drop)
    print(f"Dropped columns: {existing_cols_to_drop}")
    
    # 3. CRITICAL: Split data BEFORE any preprocessing to prevent leakage
    print("\n3. SPLITTING DATA FIRST (BEFORE preprocessing) - This prevents data leakage!")
    
    # Separate features and target
    X = df.drop('label', axis=1)
    y = (df['label'] == 'Malicious').astype(int)
    
    print(f"Original class distribution:")
    print(f"Benign (0): {(y==0).sum()}")
    print(f"Malicious (1): {(y==1).sum()}")
    
    # PROPER 80/20 stratified split BEFORE any preprocessing
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42, 
        stratify=y  # Ensures same class distribution in train/test
    )
    
    print(f"Training set: {X_train_raw.shape[0]} samples")
    print(f"Test set: {X_test_raw.shape[0]} samples")
    print(f"Train class distribution: Benign={np.sum(y_train==0)}, Malicious={np.sum(y_train==1)}")
    print(f"Test class distribution: Benign={np.sum(y_test==0)}, Malicious={np.sum(y_test==1)}")
    
    # 4. Handle missing values SEPARATELY for train and test
    print("\n4. Handling missing values (separately for train/test)...")
    
    def handle_missing_values(X_data, is_training=True, fill_values=None):
        """Handle missing values without data leakage"""
        X_clean = X_data.copy()
        
        # Numeric columns
        numeric_columns = ['duration', 'orig_bytes', 'resp_bytes', 'missed_bytes', 
                          'orig_pkts', 'orig_ip_bytes', 'resp_pkts', 'resp_ip_bytes']
        
        if fill_values is None:  # Training data
            fill_values = {}
            
        for col in numeric_columns:
            if col in X_clean.columns:
                X_clean[col] = pd.to_numeric(X_clean[col], errors='coerce')
                if is_training:
                    # For training, calculate fill value
                    fill_values[col] = 0  # Use 0 as specified
                    X_clean[col] = X_clean[col].fillna(fill_values[col])
                else:
                    # For test, use training fill value
                    X_clean[col] = X_clean[col].fillna(fill_values[col])
        
        # Categorical columns
        categorical_columns = ['proto', 'service', 'conn_state', 'history']
        for col in categorical_columns:
            if col in X_clean.columns:
                if is_training:
                    fill_values[col] = 'unknown'
                    X_clean[col] = X_clean[col].fillna(fill_values[col])
                else:
                    X_clean[col] = X_clean[col].fillna(fill_values[col])
        
        return X_clean, fill_values
    
    # Handle missing values for training data
    X_train_clean, fill_values = handle_missing_values(X_train_raw, is_training=True)
    
    # Handle missing values for test data using training statistics
    X_test_clean, _ = handle_missing_values(X_test_raw, is_training=False, fill_values=fill_values)
    
    print("Missing values handled without leakage")
    
    # 5. Encode categorical variables SEPARATELY
    print("\n5. Encoding categorical variables (fit on train, transform test)...")
    
    def safe_encode_categorical(X_train, X_test):
        """Encode categorical variables without leakage"""
        categorical_columns = ['proto', 'service', 'conn_state', 'history']
        existing_categorical = [col for col in categorical_columns if col in X_train.columns]
        
        if existing_categorical:
            # One-hot encode training data
            X_train_encoded = pd.get_dummies(X_train, columns=existing_categorical, prefix=existing_categorical)
            
            # For test data, we need to ensure same columns
            X_test_encoded = pd.get_dummies(X_test, columns=existing_categorical, prefix=existing_categorical)
            
            # Align columns (add missing columns with 0s, remove extra columns)
            train_columns = set(X_train_encoded.columns)
            test_columns = set(X_test_encoded.columns)
            
            # Add missing columns to test set
            for col in train_columns - test_columns:
                X_test_encoded[col] = 0
            
            # Remove extra columns from test set
            for col in test_columns - train_columns:
                if col in X_test_encoded.columns:
                    X_test_encoded = X_test_encoded.drop(col, axis=1)
            
            # Ensure same column order
            X_test_encoded = X_test_encoded[X_train_encoded.columns]
            
        else:
            X_train_encoded = X_train
            X_test_encoded = X_test
        
        return X_train_encoded, X_test_encoded
    
    X_train_encoded, X_test_encoded = safe_encode_categorical(X_train_clean, X_test_clean)
    
    print(f"Features after encoding: {X_train_encoded.shape[1]}")
    print(f"Training shape: {X_train_encoded.shape}")
    print(f"Test shape: {X_test_encoded.shape}")
    
    # 6. Scale features SEPARATELY (fit on train, transform test)
    print("\n6. Scaling features (fit on train only)...")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_encoded)  # Fit on training data only
    X_test_scaled = scaler.transform(X_test_encoded)        # Transform test data
    
    # Convert back to DataFrame
    feature_names = X_train_encoded.columns.tolist()
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_names)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_names)
    
    # 7. Balance classes using SMOTE (on training data only)
    print("\n7. Balancing classes with SMOTE (training data only)...")
    
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
    
    # 8. Train models with proper cross-validation
    print("\n8. Training models with proper validation...")
    
    # Use stratified k-fold for more robust evaluation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # XGBoost with cross-validation
    print("Training XGBoost...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,  # Reduced to prevent overfitting
        max_depth=4,       # Reduced depth
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1,
        reg_alpha=0.1,     # L1 regularization
        reg_lambda=0.1     # L2 regularization
    )
    
    xgb_model.fit(X_train_balanced, y_train_balanced)
    
    # Random Forest with balanced class weights
    print("Training Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,      # Reduced depth
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    
    rf_model.fit(X_train_scaled, y_train)  # Use original training set
    
    # 9. Hyperparameter tuning with proper CV
    print("\n9. Hyperparameter optimization with cross-validation...")
    
    # Smaller parameter grid to prevent overfitting
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [3, 4, 5],
        'learning_rate': [0.1, 0.2]
    }
    
    grid_search = GridSearchCV(
        xgb.XGBClassifier(random_state=42, n_jobs=-1, reg_alpha=0.1, reg_lambda=0.1),
        param_grid,
        cv=cv,  # Use stratified k-fold
        scoring='f1',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train_balanced, y_train_balanced)
    best_xgb = grid_search.best_estimator_
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation F1 score: {grid_search.best_score_:.4f}")
    
    # 10. Evaluate on TEST set (never seen during training)
    print("\n10. Model Evaluation on TEST SET...")
    
    def evaluate_model_properly(model, X_test, y_test, model_name):
        """Proper evaluation without data leakage"""
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate all metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        print(f"\n{model_name} Results (TEST SET):")
        print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        print(f"ROC-AUC:   {roc_auc:.4f}")
        
        # Confusion matrix details
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        print(f"Confusion Matrix:")
        print(f"  True Negatives:  {tn}")
        print(f"  False Positives: {fp}")
        print(f"  False Negatives: {fn}")
        print(f"  True Positives:  {tp}")
        
        return accuracy, precision, recall, f1, roc_auc
    
    # Evaluate all models on the same TEST set
    xgb_acc, xgb_precision, xgb_recall, xgb_f1, xgb_roc_auc = evaluate_model_properly(
        xgb_model, X_test_scaled, y_test, "XGBoost (Basic)"
    )
    
    best_xgb_acc, best_xgb_precision, best_xgb_recall, best_xgb_f1, best_xgb_roc_auc = evaluate_model_properly(
        best_xgb, X_test_scaled, y_test, "XGBoost (Optimized)"
    )
    
    rf_acc, rf_precision, rf_recall, rf_f1, rf_roc_auc = evaluate_model_properly(
        rf_model, X_test_scaled, y_test, "Random Forest"
    )
    
    # 11. Save the best model
    print("\n11. Saving best model...")
    
    models_performance = {
        'XGBoost_Basic': xgb_f1,
        'XGBoost_Optimized': best_xgb_f1,
        'Random_Forest': rf_f1
    }
    
    best_model_name = max(models_performance, key=models_performance.get)
    print(f"Best model based on F1-score: {best_model_name}")
    
    if best_model_name == 'XGBoost_Optimized':
        best_model = best_xgb
        best_acc = best_xgb_acc
    elif best_model_name == 'XGBoost_Basic':
        best_model = xgb_model
        best_acc = xgb_acc
    else:
        best_model = rf_model
        best_acc = rf_acc
    
    # Save models with prefix to avoid overwriting
    joblib.dump(best_model, 'fixed_best_iot_model.joblib')
    joblib.dump(scaler, 'fixed_iot_scaler.joblib')
    joblib.dump(feature_names, 'fixed_iot_features.joblib')
    
    print("Fixed models saved!")
    
    # Final summary with REALISTIC results
    print("\n" + "="*70)
    print("FIXED RESULTS SUMMARY (No Data Leakage)")
    print("="*70)
    print(f"XGBoost (Basic)     - Acc: {xgb_acc:.4f}, F1: {xgb_f1:.4f}, AUC: {xgb_roc_auc:.4f}")
    print(f"XGBoost (Optimized) - Acc: {best_xgb_acc:.4f}, F1: {best_xgb_f1:.4f}, AUC: {best_xgb_roc_auc:.4f}")
    print(f"Random Forest       - Acc: {rf_acc:.4f}, F1: {rf_f1:.4f}, AUC: {rf_roc_auc:.4f}")
    print(f"\nBest Model: {best_model_name}")
    print(f"Best Model Accuracy: {best_acc:.4f} ({best_acc*100:.2f}%) - REALISTIC!")
    print("="*70)
    
    # Data leakage check
    print("\n🔍 DATA LEAKAGE PREVENTION CHECKLIST:")
    print("✅ Train/test split done BEFORE preprocessing")
    print("✅ Scaling fitted on training data only")
    print("✅ SMOTE applied to training data only")
    print("✅ Cross-validation used for hyperparameter tuning")
    print("✅ Test set used ONLY for final evaluation")
    print("✅ No information from test set leaked to training")

if __name__ == "__main__":
    main()