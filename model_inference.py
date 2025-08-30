"""
Model Inference Script for IoT Ransomware Detection
Use this script to make predictions with the trained model
"""

import pandas as pd
import numpy as np
import joblib

class IoTRansomwarePredictor:
    """
    Class for making predictions using the trained IoT ransomware detection model
    """
    
    def __init__(self, model_path='best_iot_ransomware_model.joblib', 
                 scaler_path='iot_ransomware_scaler.joblib',
                 features_path='iot_ransomware_features.joblib'):
        """
        Initialize the predictor with saved model components
        
        Args:
            model_path: Path to saved model
            scaler_path: Path to saved scaler
            features_path: Path to saved feature names
        """
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.feature_names = joblib.load(features_path)
        print("Model loaded successfully!")
        print(f"Expected features: {len(self.feature_names)}")
    
    def preprocess_data(self, df):
        """
        Preprocess new data to match training format
        
        Args:
            df: DataFrame with new network connection data
            
        Returns:
            Preprocessed DataFrame ready for prediction
        """
        # Drop ID columns if they exist
        columns_to_drop = ['Unnamed: 0', 'id.orig_h', 'id.resp_h']
        existing_cols_to_drop = [col for col in columns_to_drop if col in df.columns]
        if existing_cols_to_drop:
            df = df.drop(columns=existing_cols_to_drop)
        
        # Remove label column if it exists (for inference)
        if 'label' in df.columns:
            df = df.drop('label', axis=1)
        
        # Handle missing values for numeric columns
        numeric_columns = ['duration', 'orig_bytes', 'resp_bytes', 'missed_bytes', 
                          'orig_pkts', 'orig_ip_bytes', 'resp_pkts', 'resp_ip_bytes']
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col] = df[col].fillna(0)
        
        # Handle missing values for categorical columns
        categorical_columns = ['proto', 'service', 'conn_state', 'history']
        for col in categorical_columns:
            if col in df.columns:
                df[col] = df[col].fillna('unknown')
        
        # One-hot encode categorical variables
        existing_categorical = [col for col in categorical_columns if col in df.columns]
        if existing_categorical:
            df_encoded = pd.get_dummies(df, columns=existing_categorical, prefix=existing_categorical)
        else:
            df_encoded = df
        
        # Ensure all expected features are present
        for feature in self.feature_names:
            if feature not in df_encoded.columns:
                df_encoded[feature] = 0
        
        # Keep only the features used in training (in the same order)
        df_final = df_encoded[self.feature_names]
        
        return df_final
    
    def predict(self, df):
        """
        Make predictions on new data
        
        Args:
            df: DataFrame with network connection data
            
        Returns:
            Predictions and probabilities
        """
        # Preprocess data
        df_processed = self.preprocess_data(df)
        
        # Scale features
        df_scaled = self.scaler.transform(df_processed)
        
        # Make predictions
        predictions = self.model.predict(df_scaled)
        probabilities = self.model.predict_proba(df_scaled)
        
        # Convert to readable format
        prediction_labels = ['Benign' if pred == 0 else 'Malicious' for pred in predictions]
        
        return prediction_labels, probabilities
    
    def predict_single_connection(self, connection_data):
        """
        Predict for a single network connection
        
        Args:
            connection_data: Dictionary with connection features
            
        Returns:
            Prediction and confidence
        """
        # Convert to DataFrame
        df = pd.DataFrame([connection_data])
        
        # Make prediction
        predictions, probabilities = self.predict(df)
        
        prediction = predictions[0]
        confidence = max(probabilities[0])
        
        return prediction, confidence

def demonstrate_usage():
    """
    Demonstrate how to use the model for inference
    """
    print("IoT Ransomware Detection - Model Inference Demo")
    print("=" * 50)
    
    try:
        # Load the predictor
        predictor = IoTRansomwarePredictor()
        
        # Example 1: Predict for sample benign connection
        print("\nExample 1: Benign HTTP connection")
        benign_connection = {
            'id.orig_p': 45123,
            'id.resp_p': 80,
            'proto': 'tcp',
            'service': 'http',
            'duration': 1.5,
            'orig_bytes': 512,
            'resp_bytes': 8192,
            'conn_state': 'SF',
            'missed_bytes': 0,
            'history': 'ShADadfF',
            'orig_pkts': 10,
            'orig_ip_bytes': 1024,
            'resp_pkts': 8,
            'resp_ip_bytes': 8704
        }
        
        prediction, confidence = predictor.predict_single_connection(benign_connection)
        print(f"Prediction: {prediction}")
        print(f"Confidence: {confidence:.4f}")
        
        # Example 2: Predict for sample malicious connection (IRC on port 6667)
        print("\nExample 2: Suspicious IRC connection")
        malicious_connection = {
            'id.orig_p': 49123,
            'id.resp_p': 6667,
            'proto': 'tcp',
            'service': 'irc',
            'duration': 2.5,
            'orig_bytes': 75,
            'resp_bytes': 243,
            'conn_state': 'S3',
            'missed_bytes': 0,
            'history': 'ShAdDaf',
            'orig_pkts': 7,
            'orig_ip_bytes': 447,
            'resp_pkts': 6,
            'resp_ip_bytes': 563
        }
        
        prediction, confidence = predictor.predict_single_connection(malicious_connection)
        print(f"Prediction: {prediction}")
        print(f"Confidence: {confidence:.4f}")
        
        # Example 3: Load and predict on a CSV file (if available)
        print("\nExample 3: Batch prediction on test data")
        try:
            # Try to load some test data
            test_data = pd.read_csv(r"c:\Users\MADHURIMA\Desktop\ml\dataset\CTU-IoT-ramsomware -Capture-1-1conn.log.labeled.csv")
            
            # Take a small sample for demonstration
            sample_data = test_data.head(10).copy()
            
            # Remove label for inference
            sample_features = sample_data.drop('label', axis=1) if 'label' in sample_data.columns else sample_data
            
            predictions, probabilities = predictor.predict(sample_features)
            
            print("Sample predictions:")
            for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
                confidence = max(prob)
                print(f"Connection {i+1}: {pred} (confidence: {confidence:.4f})")
                
        except FileNotFoundError:
            print("Test data file not found. Skipping batch prediction demo.")
        
    except FileNotFoundError as e:
        print(f"Error: Model files not found. Please run the training script first.")
        print(f"Missing file: {e}")
    except Exception as e:
        print(f"Error during inference: {e}")

def main():
    """
    Main function for inference demo
    """
    demonstrate_usage()

if __name__ == "__main__":
    main()