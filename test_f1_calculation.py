#!/usr/bin/env python3
"""
Test script to calculate F1 scores from the actual saved confusion matrix
This shows what you'll see in the backend when making predictions
"""

def calculate_f1_from_confusion_matrix():
    """Calculate F1 scores from the actual saved confusion matrix"""
    
    # Using the actual values from your backend output
    tn, fp, fn, tp = 1381, 618, 545, 1456
    
    print("F1 SCORE CALCULATION FROM SAVED CONFUSION MATRIX")
    print("=" * 60)
    print(f"Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print()
    
    # Calculate precision, recall, and F1 for each class
    precision_0 = tn / (tn + fn) if (tn + fn) > 0 else 0  # Precision for healthy class
    recall_0 = tn / (tn + fp) if (tn + fp) > 0 else 0      # Recall for healthy class
    f1_0 = 2 * (precision_0 * recall_0) / (precision_0 + recall_0) if (precision_0 + recall_0) > 0 else 0
    
    precision_1 = tp / (tp + fp) if (tp + fp) > 0 else 0   # Precision for unhealthy class
    recall_1 = tp / (tp + fn) if (tp + fn) > 0 else 0       # Recall for unhealthy class
    f1_1 = 2 * (precision_1 * recall_1) / (precision_1 + recall_1) if (precision_1 + recall_1) > 0 else 0
    
    # Calculate macro and weighted averages
    f1_macro = (f1_0 + f1_1) / 2
    total_samples = tn + fp + fn + tp
    f1_weighted = (f1_0 * (tn + fp) + f1_1 * (tp + fn)) / total_samples if total_samples > 0 else 0
    
    print("DETAILED CALCULATIONS:")
    print(f"Healthy Class (0):")
    print(f"  Precision = TN/(TN+FN) = {tn}/({tn}+{fn}) = {precision_0:.4f}")
    print(f"  Recall = TN/(TN+FP) = {tn}/({tn}+{fp}) = {recall_0:.4f}")
    print(f"  F1 = 2*(Precision*Recall)/(Precision+Recall) = {f1_0:.4f}")
    print()
    print(f"Unhealthy Class (1):")
    print(f"  Precision = TP/(TP+FP) = {tp}/({tp}+{fp}) = {precision_1:.4f}")
    print(f"  Recall = TP/(TP+FN) = {tp}/({tp}+{fn}) = {recall_1:.4f}")
    print(f"  F1 = 2*(Precision*Recall)/(Precision+Recall) = {f1_1:.4f}")
    print()
    print("FINAL F1 SCORES:")
    print(f"F1 Score (Healthy Class 0): {f1_0:.4f}")
    print(f"F1 Score (Unhealthy Class 1): {f1_1:.4f}")
    print(f"F1 Score (Macro Average): {f1_macro:.4f}")
    print(f"F1 Score (Weighted Average): {f1_weighted:.4f}")
    print("=" * 60)
    
    print("\nWHAT YOU'LL SEE IN YOUR BACKEND CONSOLE:")
    print("=" * 60)
    print("Saved Confusion Matrix: TN=1381, FP=618, FN=545, TP=1456")
    print()
    print("F1 SCORES (Calculated from Saved Confusion Matrix):")
    print(f"F1 Score (Healthy Class 0): {f1_0:.4f}")
    print(f"F1 Score (Unhealthy Class 1): {f1_1:.4f}")
    print(f"F1 Score (Macro Average): {f1_macro:.4f}")
    print(f"F1 Score (Weighted Average): {f1_weighted:.4f}")
    print("=" * 60)

if __name__ == "__main__":
    calculate_f1_from_confusion_matrix() 