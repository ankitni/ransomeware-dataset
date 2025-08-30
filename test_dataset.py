"""
Quick test to verify dataset loading and basic info
"""

import pandas as pd
import numpy as np

def test_dataset():
    try:
        # Load dataset
        df = pd.read_csv(r"c:\Users\MADHURIMA\Desktop\ml\dataset\CTU-IoT-ramsomware -Capture-1-1conn.log.labeled.csv")
        
        print("✓ Dataset loaded successfully!")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"Missing values: {df.isnull().sum().sum()}")
        print(f"Label distribution:\n{df['label'].value_counts()}")
        
        # Check data types
        print(f"\nData types:")
        print(df.dtypes)
        
        return True
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        return False

if __name__ == "__main__":
    test_dataset()