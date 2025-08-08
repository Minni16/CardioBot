import joblib
import json
import numpy as np

# Load the model and metrics
with open('patient_model_metrics.json', 'r') as f:
    metrics = json.load(f)

loaded_data = joblib.load('patient_model.pkl')
model = loaded_data['model']
scaler = loaded_data['scaler']

feature_names = metrics['feature_names']
coefficients = model.coef_[0]

print("=== Model Coefficients Analysis ===")
print("Feature Name -> Coefficient -> Interpretation")
print("-" * 50)

for i, feature_name in enumerate(feature_names):
    coef = coefficients[i]
    if abs(coef) < 0.01:
        interpretation = "❌ NEAR ZERO - Model didn't learn this feature"
    elif coef > 0.1:
        interpretation = "✅ STRONG POSITIVE - Increases risk"
    elif coef > 0.01:
        interpretation = "✅ MODERATE POSITIVE - Increases risk"
    elif coef < -0.1:
        interpretation = "✅ STRONG NEGATIVE - Decreases risk"
    elif coef < -0.01:
        interpretation = "✅ MODERATE NEGATIVE - Decreases risk"
    else:
        interpretation = "⚠️ WEAK - Minimal impact"
    
    print(f"{feature_name:20} -> {coef:8.4f} -> {interpretation}")

print("\n=== Medical Conditions Analysis ===")
medical_conditions = ['High_Blood_Pressure', 'Diabetes', 'High_Cholesterol', 'Family_History']
for condition in medical_conditions:
    if condition in feature_names:
        idx = feature_names.index(condition)
        coef = coefficients[idx]
        print(f"{condition}: {coef:.4f} {'❌' if abs(coef) < 0.01 else '✅'}") 