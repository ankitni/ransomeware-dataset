"""
IoT Ransomware Detection using Machine Learning
Dataset: CTU-IoT-Malware Capture Project

This script implements a complete ML pipeline for detecting malicious network traffic
in IoT devices, specifically focusing on ransomware detection.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class IoTRansomwareDetector:
    """
    A comprehensive machine learning pipeline for IoT ransomware detection
    """
    
    def __init__(self, data_path):
        """
        Initialize the detector with dataset path
        
        Args:
            data_path (str): Path to the CSV dataset
        """
        self.data_path = data_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = None
        self.model = None
        self.feature_names = None
        
    def load_and_explore_data(self):
        """
        Load the dataset and perform initial exploration
        """
        print("Loading dataset...")
        self.df = pd.read_csv(self.data_path)
        
        print(f"Dataset shape: {self.df.shape}")
        print(f"Columns: {list(self.df.columns)}")
        print("\nFirst few rows:")
        print(self.df.head())
        
        print("\nDataset info:")
        print(self.df.info())
        
        print("\nMissing values:")
        print(self.df.isnull().sum())
        
        print("\nClass distribution:")
        print(self.df['label'].value_counts())
        
        # Visualize class distribution
        plt.figure(figsize=(8, 6))
        self.df['label'].value_counts().plot(kind='bar')
        plt.title('Class Distribution')
        plt.xlabel('Label')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('class_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return self.df
    
    def preprocess_data(self):
        """
        Clean and preprocess the dataset
        """
        print("\nPreprocessing data...")
        
        # Drop useless columns
        columns_to_drop = ['Unnamed: 0', 'id.orig_h', 'id.resp_h']
        existing_cols_to_drop = [col for col in columns_to_drop if col in self.df.columns]
        self.df = self.df.drop(columns=existing_cols_to_drop)
        print(f"Dropped columns: {existing_cols_to_drop}")
        
        # Handle missing values
        numeric_columns = ['duration', 'orig_bytes', 'resp_bytes', 'missed_bytes', 
                          'orig_pkts', 'orig_ip_bytes', 'resp_pkts', 'resp_ip_bytes']
        
        for col in numeric_columns:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
                self.df[col] = self.df[col].fillna(0)
        
        # Handle categorical missing values
        categorical_columns = ['proto', 'service', 'conn_state', 'history']
        for col in categorical_columns:
            if col in self.df.columns:
                self.df[col] = self.df[col].fillna('unknown')
        
        print("Missing values handled")
        print(f"Remaining missing values: {self.df.isnull().sum().sum()}")
        
        return self.df
    
    def encode_features(self):
        """
        Encode categorical variables using one-hot encoding
        """
        print("\nEncoding categorical features...")
        
        # Separate features and target
        X = self.df.drop('label', axis=1)
        y = self.df['label']
        
        # Encode target variable
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        
        # One-hot encode categorical variables
        categorical_columns = ['proto', 'service', 'conn_state', 'history']
        existing_categorical = [col for col in categorical_columns if col in X.columns]
        
        if existing_categorical:
            X_encoded = pd.get_dummies(X, columns=existing_categorical, prefix=existing_categorical)
        else:
            X_encoded = X
        
        self.feature_names = X_encoded.columns.tolist()
        
        print(f"Features after encoding: {X_encoded.shape[1]}")
        print(f"Target classes: {le.classes_}")
        
        return X_encoded, y_encoded, le
    
    def scale_features(self, X_train, X_test):
        """
        Scale numeric features using StandardScaler
        
        Args:
            X_train: Training features
            X_test: Test features
            
        Returns:
            Scaled training and test features
        """
        print("\nScaling features...")
        
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Convert back to DataFrame to maintain feature names
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=self.feature_names)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=self.feature_names)
        
        return X_train_scaled, X_test_scaled
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """
        Split data into train and test sets with stratification
        
        Args:
            X: Features
            y: Target
            test_size: Proportion of test set
            random_state: Random seed
            
        Returns:
            Train/test split
        """
        print(f"\nSplitting data (train: {1-test_size:.0%}, test: {test_size:.0%})...")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
        return X_train, X_test, y_train, y_test
    
    def balance_classes_smote(self, X_train, y_train):
        """
        Balance classes using SMOTE oversampling
        
        Args:
            X_train: Training features
            y_train: Training labels
            
        Returns:
            Balanced training set
        """
        print("\nBalancing classes with SMOTE...")
        
        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        
        unique, counts = np.unique(y_train_balanced, return_counts=True)
        print("Class distribution after SMOTE:")
        for class_val, count in zip(unique, counts):
            print(f"Class {class_val}: {count} samples")
        
        return X_train_balanced, y_train_balanced
    
    def train_random_forest(self, X_train, y_train, use_class_weights=True):
        """
        Train Random Forest model with hyperparameter tuning
        
        Args:
            X_train: Training features
            y_train: Training labels
            use_class_weights: Whether to use class weights for balancing
            
        Returns:
            Trained Random Forest model
        """
        print("\nTraining Random Forest model...")
        
        # Set class weights if specified
        if use_class_weights:
            class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
            class_weight_dict = dict(zip(np.unique(y_train), class_weights))
            print(f"Using class weights: {class_weight_dict}")
        else:
            class_weight_dict = None
        
        # Define parameter grid for tuning
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2']
        }
        
        # Initialize Random Forest
        rf = RandomForestClassifier(
            random_state=42,
            class_weight=class_weight_dict,
            n_jobs=-1
        )
        
        # Perform randomized search for faster tuning
        rf_random = RandomizedSearchCV(
            rf, param_grid, n_iter=20, cv=3, random_state=42, 
            n_jobs=-1, scoring='f1_macro', verbose=1
        )
        
        rf_random.fit(X_train, y_train)
        
        print(f"Best parameters: {rf_random.best_params_}")
        print(f"Best cross-validation score: {rf_random.best_score_:.4f}")
        
        return rf_random.best_estimator_
    
    def train_xgboost(self, X_train, y_train):
        """
        Train XGBoost model with hyperparameter tuning
        
        Args:
            X_train: Training features
            y_train: Training labels
            
        Returns:
            Trained XGBoost model
        """
        print("\nTraining XGBoost model...")
        
        # Calculate scale_pos_weight for class balancing
        neg_count = np.sum(y_train == 0)
        pos_count = np.sum(y_train == 1)
        scale_pos_weight = neg_count / pos_count
        
        print(f"Scale pos weight: {scale_pos_weight:.2f}")
        
        # Define parameter grid
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0]
        }
        
        # Initialize XGBoost
        xgb_model = xgb.XGBClassifier(
            random_state=42,
            scale_pos_weight=scale_pos_weight,
            n_jobs=-1
        )
        
        # Perform randomized search
        xgb_random = RandomizedSearchCV(
            xgb_model, param_grid, n_iter=20, cv=3, random_state=42,
            n_jobs=-1, scoring='f1_macro', verbose=1
        )
        
        xgb_random.fit(X_train, y_train)
        
        print(f"Best parameters: {xgb_random.best_params_}")
        print(f"Best cross-validation score: {xgb_random.best_score_:.4f}")
        
        return xgb_random.best_estimator_
    
    def evaluate_model(self, model, X_test, y_test, model_name="Model"):
        """
        Comprehensive model evaluation
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            model_name: Name for the model
        """
        print(f"\nEvaluating {model_name}...")
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Classification report
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        
        # ROC-AUC Score
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        print(f"ROC-AUC Score: {roc_auc:.4f}")
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Benign', 'Malicious'], 
                   yticklabels=['Benign', 'Malicious'])
        plt.title(f'{model_name} - Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(f'{model_name.lower().replace(" ", "_")}_confusion_matrix.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'{model_name.lower().replace(" ", "_")}_roc_curve.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        return roc_auc, y_pred, y_pred_proba
    
    def plot_feature_importance(self, model, model_name="Model", top_n=20):
        """
        Plot feature importance
        
        Args:
            model: Trained model
            model_name: Name of the model
            top_n: Number of top features to display
        """
        if hasattr(model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            plt.figure(figsize=(10, 8))
            sns.barplot(data=feature_importance.head(top_n), 
                       x='importance', y='feature')
            plt.title(f'{model_name} - Top {top_n} Feature Importance')
            plt.xlabel('Importance')
            plt.tight_layout()
            plt.savefig(f'{model_name.lower().replace(" ", "_")}_feature_importance.png', 
                       dpi=300, bbox_inches='tight')
            plt.show()
            
            return feature_importance
        else:
            print(f"{model_name} does not support feature importance")
            return None
    
    def save_model(self, model, scaler, filename):
        """
        Save trained model and scaler
        
        Args:
            model: Trained model
            scaler: Fitted scaler
            filename: Base filename for saving
        """
        print(f"\nSaving model and scaler...")
        
        # Save model
        model_filename = f"{filename}_model.joblib"
        joblib.dump(model, model_filename)
        print(f"Model saved as: {model_filename}")
        
        # Save scaler
        scaler_filename = f"{filename}_scaler.joblib"
        joblib.dump(scaler, scaler_filename)
        print(f"Scaler saved as: {scaler_filename}")
        
        # Save feature names
        feature_filename = f"{filename}_features.joblib"
        joblib.dump(self.feature_names, feature_filename)
        print(f"Feature names saved as: {feature_filename}")
    
    def load_model(self, filename):
        """
        Load saved model and scaler
        
        Args:
            filename: Base filename for loading
            
        Returns:
            Loaded model, scaler, and feature names
        """
        model = joblib.load(f"{filename}_model.joblib")
        scaler = joblib.load(f"{filename}_scaler.joblib")
        feature_names = joblib.load(f"{filename}_features.joblib")
        
        print(f"Model loaded from: {filename}")
        return model, scaler, feature_names
    
    def run_complete_pipeline(self, use_smote=True, model_type='xgboost'):
        """
        Run the complete ML pipeline
        
        Args:
            use_smote: Whether to use SMOTE for class balancing
            model_type: 'xgboost' or 'random_forest'
        """
        print("=" * 60)
        print("IoT RANSOMWARE DETECTION - COMPLETE ML PIPELINE")
        print("=" * 60)
        
        # Step 1: Load and explore data
        self.load_and_explore_data()
        
        # Step 2: Preprocess data
        self.preprocess_data()
        
        # Step 3: Encode features
        X, y, label_encoder = self.encode_features()
        
        # Step 4: Split data
        X_train, X_test, y_train, y_test = self.split_data(X, y)
        
        # Step 5: Scale features
        X_train_scaled, X_test_scaled = self.scale_features(X_train, X_test)
        
        # Step 6: Balance classes (optional)
        if use_smote:
            X_train_final, y_train_final = self.balance_classes_smote(X_train_scaled, y_train)
        else:
            X_train_final, y_train_final = X_train_scaled, y_train
        
        # Step 7: Train model
        if model_type.lower() == 'xgboost':
            model = self.train_xgboost(X_train_final, y_train_final)
            model_name = "XGBoost"
        else:
            model = self.train_random_forest(X_train_final, y_train_final, 
                                           use_class_weights=not use_smote)
            model_name = "Random Forest"
        
        # Step 8: Evaluate model
        roc_auc, y_pred, y_pred_proba = self.evaluate_model(
            model, X_test_scaled, y_test, model_name
        )
        
        # Step 9: Feature importance
        feature_importance = self.plot_feature_importance(model, model_name)
        
        # Step 10: Save model
        self.save_model(model, self.scaler, f"iot_ransomware_{model_type}")
        
        # Store results
        self.model = model
        self.X_test = X_test_scaled
        self.y_test = y_test
        
        print("\n" + "=" * 60)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print(f"Final {model_name} ROC-AUC Score: {roc_auc:.4f}")
        print("=" * 60)
        
        return model, roc_auc, feature_importance

def main():
    """
    Main function to run the IoT ransomware detection pipeline
    """
    # Dataset path
    data_path = r"c:\Users\MADHURIMA\Desktop\ml\dataset\CTU-IoT-ramsomware -Capture-1-1conn.log.labeled.csv"
    
    # Initialize detector
    detector = IoTRansomwareDetector(data_path)
    
    # Run pipeline with XGBoost
    print("Running pipeline with XGBoost...")
    xgb_model, xgb_auc, xgb_importance = detector.run_complete_pipeline(
        use_smote=True, model_type='xgboost'
    )
    
    print("\n" + "=" * 80)
    
    # Run pipeline with Random Forest for comparison
    print("Running pipeline with Random Forest...")
    detector_rf = IoTRansomwareDetector(data_path)
    rf_model, rf_auc, rf_importance = detector_rf.run_complete_pipeline(
        use_smote=False, model_type='random_forest'
    )
    
    # Compare results
    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    print(f"XGBoost ROC-AUC: {xgb_auc:.4f}")
    print(f"Random Forest ROC-AUC: {rf_auc:.4f}")
    
    if xgb_auc > rf_auc:
        print("XGBoost performs better!")
    else:
        print("Random Forest performs better!")
    
    print("=" * 60)

if __name__ == "__main__":
    main()