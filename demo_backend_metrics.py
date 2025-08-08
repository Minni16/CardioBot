#!/usr/bin/env python3
"""
Demonstration script for backend F1 score and confusion matrix printing
This script shows what you'll see in the backend console when models are trained or predictions are made.
"""

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score

def demo_metrics_printing():
    """Demonstrate the metrics printing functionality"""
    
    print("BACKEND METRICS PRINTING DEMONSTRATION")
    print("=" * 60)
    print("This is what you'll see in your backend console when:")
    print("1. Models are retrained")
    print("2. Predictions are made")
    print("3. Model evaluation is performed")
    print("=" * 60)
    
    # Simulate realistic test data (similar to your actual models)
    np.random.seed(42)
    n_samples = 4000  # Similar to your patient dataset size
    
    # Generate realistic predictions with some errors
    y_test = np.random.choice([0, 1], size=n_samples, p=[0.5, 0.5])
    y_pred = y_test.copy()
    
    # Introduce some prediction errors (similar to real model performance)
    error_indices = np.random.choice(n_samples, size=int(n_samples * 0.29), replace=False)  # ~71% accuracy
    y_pred[error_indices] = 1 - y_pred[error_indices]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    # Print detailed metrics (this is what you'll see in backend)
    print("PATIENT MODEL EVALUATION METRICS")
    print("=" * 60)
    print(f"Model Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print("\nCONFUSION MATRIX:")
    print("                 Predicted")
    print("                 Healthy | Unhealthy")
    print(f"Actual Healthy    |    {cm[0][0]:4d}    |   {cm[0][1]:4d}")
    print(f"Actual Unhealthy  |    {cm[1][0]:4d}    |   {cm[1][1]:4d}")
    print(f"\nTrue Negatives (TN): {cm[0][0]} - Correctly predicted healthy")
    print(f"False Positives (FP): {cm[0][1]} - Healthy predicted as unhealthy")
    print(f"False Negatives (FN): {cm[1][0]} - Unhealthy predicted as healthy")
    print(f"True Positives (TP): {cm[1][1]} - Correctly predicted unhealthy")
    print("\nCLASSIFICATION REPORT:")
    print(report)
    
    # Calculate and print F1 scores for each class
    f1_class_0 = f1_score(y_test, y_pred, pos_label=0)  # F1 for healthy class
    f1_class_1 = f1_score(y_test, y_pred, pos_label=1)  # F1 for unhealthy class
    f1_macro = f1_score(y_test, y_pred, average='macro')  # Macro average F1
    f1_weighted = f1_score(y_test, y_pred, average='weighted')  # Weighted average F1
    
    print("\nF1 SCORES:")
    print(f"F1 Score (Healthy Class 0): {f1_class_0:.4f}")
    print(f"F1 Score (Unhealthy Class 1): {f1_class_1:.4f}")
    print(f"F1 Score (Macro Average): {f1_macro:.4f}")
    print(f"F1 Score (Weighted Average): {f1_weighted:.4f}")
    print("=" * 60)
    
    print("\n" + "=" * 60)
    print("HOW TO USE THIS IN YOUR BACKEND:")
    print("=" * 60)
    print("1. When you retrain models, you'll see these metrics automatically")
    print("2. When making predictions, current model metrics are displayed")
    print("3. Use the utility function: print_model_metrics(y_test, y_pred, 'Model Name')")
    print("4. Run Django management command: python manage.py retrain_models")
    print("5. Check your console/terminal for detailed metrics output")
    print("=" * 60)

if __name__ == "__main__":
    demo_metrics_printing() 