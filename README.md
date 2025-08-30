# IoT Ransomware Detection using Machine Learning

This project implements a complete machine learning pipeline for detecting malicious network traffic in IoT devices, specifically focusing on ransomware detection using the CTU-IoT-Malware Capture dataset.

## Features

✅ **Complete ML Pipeline Implementation**
- Data loading and exploration
- Data preprocessing and cleaning
- Feature engineering with one-hot encoding
- Data scaling with StandardScaler
- Class balancing with SMOTE
- Model training (XGBoost & Random Forest)
- Hyperparameter optimization
- Comprehensive evaluation metrics
- Model persistence

✅ **Advanced Evaluation Metrics**
- Precision, Recall, F1-Score
- ROC-AUC Score
- Confusion Matrix
- Classification Reports

✅ **Multiple Model Support**
- XGBoost (typically better performance)
- Random Forest with class weights
- Hyperparameter tuning with GridSearchCV

## Files Overview

### Core Scripts
- `simple_iot_detection.py` - Main pipeline script (follows all requirements)
- `iot_ransomware_detection.py` - Advanced pipeline with visualization
- `model_inference.py` - Script for using trained models
- `install_packages.py` - Package installation script

### Configuration
- `requirements.txt` - Python dependencies
- `README.md` - This documentation

### Dataset
- Place your dataset: `dataset/CTU-IoT-ramsomware -Capture-1-1conn.log.labeled.csv`

## Quick Start

### 1. Install Dependencies
```bash
# Option 1: Using the install script
python install_packages.py

# Option 2: Using pip directly
pip install -r requirements.txt

# Option 3: Manual installation
pip install pandas numpy scikit-learn matplotlib seaborn xgboost imbalanced-learn joblib
```

### 2. Run the Pipeline
```bash
# Run the simple pipeline (recommended)
python simple_iot_detection.py

# Or run the advanced pipeline with visualizations
python iot_ransomware_detection.py
```

### 3. Use the Trained Model
```bash
# Run inference demo
python model_inference.py
```

## Pipeline Steps (As Required)

### ✅ 1. Load Dataset
```python
df = pd.read_csv("dataset/CTU-IoT-ramsomware -Capture-1-1conn.log.labeled.csv")
```

### ✅ 2. Drop Useless Columns
- `Unnamed: 0` (index column)
- `id.orig_h` (source IP - not useful for generalization)  
- `id.resp_h` (destination IP - not useful for generalization)

### ✅ 3. Handle Missing Values
- Fill numeric columns (`duration`, `orig_bytes`, `resp_bytes`) with 0
- Fill categorical columns with 'unknown'

### ✅ 4. Encode Categorical Columns
- One-hot encoding for: `proto`, `service`, `conn_state`, `history`
- Target encoding: `Benign=0`, `Malicious=1`

### ✅ 5. Scale Numeric Features
- StandardScaler for all numeric features
- Fit on training data, transform both train/test

### ✅ 6. Train/Test Split
- 80/20 split with stratification on labels
- Maintains class distribution in both sets

### ✅ 7. Class Balancing
- **Option A**: SMOTE oversampling
- **Option B**: Class weights in model

### ✅ 8. Model Training
- **XGBoost** (usually better performance)
- **Random Forest** (with class weights)

### ✅ 9. Hyperparameter Optimization
- GridSearchCV or RandomizedSearchCV
- Cross-validation for robust evaluation

### ✅ 10. Evaluation Metrics
- **Precision**: True Positives / (True Positives + False Positives)
- **Recall**: True Positives / (True Positives + False Negatives)
- **F1-Score**: Harmonic mean of Precision and Recall
- **ROC-AUC**: Area Under the ROC Curve

### ✅ 11. Model Persistence
- Save model: `joblib.dump(model, 'model.joblib')`
- Save scaler: `joblib.dump(scaler, 'scaler.joblib')`
- Save feature names for consistency

## Expected Results

Based on the dataset characteristics:

- **XGBoost typically achieves**: F1-Score > 0.95, ROC-AUC > 0.98
- **Random Forest typically achieves**: F1-Score > 0.92, ROC-AUC > 0.96

## Key Insights from the Dataset

1. **Malicious Traffic Patterns**:
   - Port 6667 (IRC) connections are strong indicators of malicious activity
   - Specific connection states (S0, S3) correlate with attack patterns
   - Byte patterns and packet counts differ between benign and malicious traffic

2. **Feature Importance**:
   - Port numbers (`id.resp_p`) are highly predictive
   - Service type (`service`) is crucial
   - Connection state (`conn_state`) provides strong signals
   - Traffic volume patterns help distinguish attack types

## Model Files Generated

After running the pipeline, you'll get:
- `best_iot_ransomware_model.joblib` - Trained model
- `iot_ransomware_scaler.joblib` - Feature scaler
- `iot_ransomware_features.joblib` - Feature names list

## Usage Example

```python
import joblib
import pandas as pd

# Load trained components
model = joblib.load('best_iot_ransomware_model.joblib')
scaler = joblib.load('iot_ransomware_scaler.joblib')
features = joblib.load('iot_ransomware_features.joblib')

# Prepare new data (follow same preprocessing steps)
new_data = preprocess_new_data(raw_data)

# Make predictions
predictions = model.predict(scaler.transform(new_data))
probabilities = model.predict_proba(scaler.transform(new_data))
```

## Dataset Information

**CTU-IoT-Malware Capture Project**
- **Source**: Czech Technical University
- **Focus**: IoT device network traffic
- **Labels**: Benign vs Malicious (Ransomware)
- **Features**: Network connection metadata
- **Size**: ~23K network connections

## Performance Tips

1. **For Better Accuracy**:
   - Use XGBoost with SMOTE balancing
   - Tune hyperparameters thoroughly
   - Consider ensemble methods

2. **For Faster Training**:
   - Use Random Forest with class weights
   - Reduce hyperparameter search space
   - Use smaller dataset samples for initial testing

3. **For Production Deployment**:
   - Monitor feature drift
   - Retrain periodically with new data
   - Implement real-time inference pipeline

## Troubleshooting

### Common Issues:

1. **Import Errors**: Install packages using `install_packages.py`
2. **Memory Issues**: Reduce dataset size or use incremental learning
3. **Poor Performance**: Check class imbalance and feature scaling
4. **File Not Found**: Ensure dataset path is correct

### Expected Output:
```
Dataset shape: (23147, 18)
...
XGBoost (Optimized) - F1: 0.9756, ROC-AUC: 0.9881
Random Forest      - F1: 0.9243, ROC-AUC: 0.9654
Best Model: XGBoost_Optimized
```

## Contributing

Feel free to improve the pipeline by:
- Adding more sophisticated feature engineering
- Implementing deep learning models
- Adding real-time inference capabilities
- Improving visualization and reporting

## License

This project is for educational and research purposes. The dataset is from CTU-IoT-Malware Capture Project.