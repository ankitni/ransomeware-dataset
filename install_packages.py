"""
Script to install required packages for IoT Ransomware Detection
"""

import subprocess
import sys

def install_package(package):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✓ Successfully installed {package}")
    except subprocess.CalledProcessError:
        print(f"✗ Failed to install {package}")

def main():
    packages = [
        "pandas>=1.5.0",
        "numpy>=1.21.0", 
        "scikit-learn>=1.1.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "xgboost>=1.6.0",
        "imbalanced-learn>=0.9.0",
        "joblib>=1.1.0"
    ]
    
    print("Installing required packages for IoT Ransomware Detection...")
    print("=" * 60)
    
    for package in packages:
        install_package(package)
    
    print("=" * 60)
    print("Installation completed!")
    print("\nYou can now run:")
    print("python simple_iot_detection.py")

if __name__ == "__main__":
    main()