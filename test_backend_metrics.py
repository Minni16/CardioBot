#!/usr/bin/env python3
"""
Test script to demonstrate backend F1 score and confusion matrix printing
"""

import os
import sys
import django

# Add the project directory to Python path
sys.path.append('/c%3A/xampp/htdocs/Heart-Disease-Prediction-System-New')

# Set up Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'health_desease.settings')
django.setup()

from health.views import print_model_metrics
import numpy as np
from sklearn.metrics import f1_score

def test_metrics_printing():
    """Test the metrics printing functionality"""
    
    print("Testing Backend Metrics Printing Functionality")
    print("=" * 60)
    
    # Simulate some test data
    y_test = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    
    # Test the utility function
    metrics = print_model_metrics(y_test, y_pred, "TEST MODEL")
    
    print("\n" + "=" * 60)
    print("RETURNED METRICS DICTIONARY:")
    print("=" * 60)
    for key, value in metrics.items():
        if key == 'confusion_matrix':
            print(f"{key}:")
            print(value)
        elif key == 'f1_scores':
            print(f"{key}:")
            for f1_key, f1_value in value.items():
                print(f"  {f1_key}: {f1_value:.4f}")
        else:
            print(f"{key}: {value}")
    
    print("\n" + "=" * 60)
    print("TEST COMPLETED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    test_metrics_printing() 