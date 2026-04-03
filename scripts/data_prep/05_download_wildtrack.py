"""
05_download_wildtrack.py — Download the WILDTRACK dataset from Kaggle.

Requires: pip install kagglehub

Usage:
    python scripts/data_prep/05_download_wildtrack.py
"""

import kagglehub

path = kagglehub.dataset_download("aryashah2k/large-scale-multicamera-detection-dataset")

print(f"\nPath to dataset files: {path}")
