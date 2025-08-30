"""
Script to get the actual detection accuracy and performance metrics
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from imblearn.over_sampling import SMOTE

def get_model_performance():
    """Get actual performance metrics from the trained model"""
    
    print("🎯 IoT RANSOMWARE DETECTION - ACTUAL PERFORMANCE RESULTS")
    print("=" * 70)
    
    # Load the dataset
    print("Loading dataset and preprocessing...")
    df = pd.read_csv(r"dataset\CTU-IoT-ramsomware -Capture-1-1conn.log.labeled.csv")
    
    # Preprocess exactly as in training
    # Drop useless columns
    columns_to_drop = ['Unnamed: 0', 'id.orig_h', 'id.resp_h']
    existing_cols_to_drop = [col for col in columns_to_drop if col in df.columns]
    df = df.drop(columns=existing_cols_to_drop)
    
    # Handle missing values
    numeric_columns = ['duration', 'orig_bytes', 'resp_bytes', 'missed_bytes', 
                      'orig_pkts', 'orig_ip_bytes', 'resp_pkts', 'resp_ip_bytes']
    
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(0)
    
    # Fill categorical missing values
    categorical_columns = ['proto', 'service', 'conn_state', 'history']
    for col in categorical_columns:
        if col in df.columns:
            df[col] = df[col].fillna('unknown')
    
    # Separate features and target
    X = df.drop('label', axis=1)
    y = (df['label'] == 'Malicious').astype(int)
    
    # One-hot encode
    categorical_columns = ['proto', 'service', 'conn_state', 'history']
    existing_categorical = [col for col in categorical_columns if col in X.columns]
    X_encoded = pd.get_dummies(X, columns=existing_categorical, prefix=existing_categorical)
    
    # Split data (same as training)
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Load the trained model
    try:
        model = joblib.load('best_iot_ransomware_model.joblib')
        print("✓ Model loaded successfully")
    except FileNotFoundError:
        print("❌ Model file not found. Please run simple_iot_detection.py first.")
        return
    
    # Make predictions on test set
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Calculate all performance metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"\n📊 DETECTION PERFORMANCE METRICS")
    print("=" * 50)
    print(f"🎯 ACCURACY:     {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"🎯 PRECISION:    {precision:.4f} ({precision*100:.2f}%)")
    print(f"🎯 RECALL:       {recall:.4f} ({recall*100:.2f}%)")
    print(f"🎯 F1-SCORE:     {f1:.4f} ({f1*100:.2f}%)")
    print(f"🎯 ROC-AUC:      {roc_auc:.4f} ({roc_auc*100:.2f}%)")
    
    # Break down by class
    print(f"\n📈 DETAILED PERFORMANCE BREAKDOWN")
    print("=" * 50)
    print(classification_report(y_test, y_pred, target_names=['Benign', 'Malicious']))
    
    # Test set statistics
    test_malicious = np.sum(y_test == 1)
    test_benign = np.sum(y_test == 0)
    total_test = len(y_test)
    
    print(f"\n📋 TEST SET STATISTICS")
    print("=" * 50)
    print(f"Total Test Samples: {total_test:,}")
    print(f"Malicious Samples: {test_malicious:,} ({test_malicious/total_test*100:.1f}%)")
    print(f"Benign Samples: {test_benign:,} ({test_benign/total_test*100:.1f}%)")
    
    # Confusion matrix breakdown
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    print(f"\n🔍 DETECTION RESULTS BREAKDOWN")
    print("=" * 50)
    print(f"True Positives (Correctly detected malicious):  {tp:,}")
    print(f"True Negatives (Correctly detected benign):     {tn:,}")
    print(f"False Positives (Benign flagged as malicious):  {fp:,}")
    print(f"False Negatives (Malicious missed):             {fn:,}")
    
    # Calculate specific rates
    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
    false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    print(f"\n⚠️  ERROR ANALYSIS")
    print("=" * 50)
    print(f"False Positive Rate: {false_positive_rate:.4f} ({false_positive_rate*100:.2f}%)")
    print(f"False Negative Rate: {false_negative_rate:.4f} ({false_negative_rate*100:.2f}%)")
    
    # Security perspective
    print(f"\n🛡️  SECURITY EFFECTIVENESS")
    print("=" * 50)
    malicious_detected = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
    malicious_missed = fn / (tp + fn) * 100 if (tp + fn) > 0 else 0
    
    print(f"Malicious Traffic Detected: {malicious_detected:.2f}%")
    print(f"Malicious Traffic Missed:   {malicious_missed:.2f}%")
    print(f"Benign Traffic Accuracy:    {tn/(tn+fp)*100:.2f}%")
    
    # Overall assessment
    print(f"\n🏆 OVERALL ASSESSMENT")
    print("=" * 50)
    if accuracy >= 0.95:
        rating = "EXCELLENT"
        emoji = "🟢"
    elif accuracy >= 0.90:
        rating = "VERY GOOD"
        emoji = "🟡"
    elif accuracy >= 0.85:
        rating = "GOOD"
        emoji = "🟠"
    else:
        rating = "NEEDS IMPROVEMENT"
        emoji = "🔴"
    
    print(f"{emoji} Detection Rating: {rating}")
    print(f"Overall Accuracy: {accuracy*100:.2f}%")
    
    return accuracy, precision, recall, f1, roc_auc

if __name__ == "__main__":
    get_model_performance()