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

# Test case: High risk patient
test_case = {
    'Age': 65,
    'Gender': 0,  # Male
    'Height': 170,
    'Weight': 85,
    'BMI': 29.4,
    'Smoke': 1,
    'Time_of_Smoking': 25,
    'Frequency_of_smoking': 4,
    'High_Blood_Pressure': 1,
    'Diabetes': 1,
    'High_Cholesterol': 1,
    'Family_History': 1,
    'Chest_Pain': 3,
    'Chest_Pain_Severity': 4,
    'Short_Breath': 3,
    'Short_Breath_Duration': 3,
    'Exercise': 0,
    'Fatty_Food': 4,
    'Stress': 4
}

# Calculate impacts using the corrected logic
coefficients = model.coef_[0]
feature_means = scaler.mean_
feature_scales = scaler.scale_

# Calculate total log-odds contribution for normalization
total_log_odds = 0
for i, feature_name in enumerate(feature_names):
    original_val = test_case.get(feature_name, 0)
    log_odds_contribution = ((original_val - feature_means[i]) / feature_scales[i]) * coefficients[i]
    total_log_odds += log_odds_contribution

print("=== Impact Analysis ===")
print(f"Total log-odds: {total_log_odds:.3f}")

for i, feature_name in enumerate(feature_names):
    original_val = test_case.get(feature_name, 0)
    # Calculate impact based on medical knowledge
    impact_score = 0.0  # Default
    
    if feature_name == 'Age':
        if original_val >= 65:
            impact_score = 0.5  # Strong risk
        elif original_val >= 50:
            impact_score = 0.3  # Moderate risk
        elif original_val >= 35:
            impact_score = 0.1  # Low risk
        else:
            impact_score = -0.1  # Protective
    elif feature_name == 'Gender':
        impact_score = 0.2 if original_val == 0 else -0.1  # Male risk, Female protective
    elif feature_name == 'BMI':
        if original_val >= 30:
            impact_score = 0.4  # Strong risk (obese)
        elif original_val >= 25:
            impact_score = 0.2  # Moderate risk (overweight)
        else:
            impact_score = -0.1  # Protective
    elif feature_name == 'Smoke':
        impact_score = 0.4 if original_val == 1 else -0.1  # Smoking risk
    elif feature_name == 'Time_of_Smoking':
        if original_val >= 20:
            impact_score = 0.5  # Strong risk
        elif original_val >= 10:
            impact_score = 0.3  # Moderate risk
        elif original_val >= 5:
            impact_score = 0.2  # Low risk
        else:
            impact_score = 0.0  # No impact
    elif feature_name == 'High_Blood_Pressure':
        impact_score = 0.4 if original_val == 1 else -0.1  # Hypertension risk
    elif feature_name == 'Diabetes':
        impact_score = 0.5 if original_val == 1 else -0.1  # Diabetes risk
    elif feature_name == 'High_Cholesterol':
        impact_score = 0.3 if original_val == 1 else -0.1  # Cholesterol risk
    elif feature_name == 'Family_History':
        impact_score = 0.3 if original_val == 1 else -0.1  # Family history risk
    elif feature_name == 'Chest_Pain':
        if original_val >= 3:
            impact_score = 0.5  # Strong risk
        elif original_val >= 2:
            impact_score = 0.3  # Moderate risk
        elif original_val >= 1:
            impact_score = 0.2  # Low risk
        else:
            impact_score = 0.0  # No impact
    elif feature_name == 'Chest_Pain_Severity':
        if original_val >= 4:
            impact_score = 0.5  # Strong risk
        elif original_val >= 3:
            impact_score = 0.4  # Moderate risk
        elif original_val >= 2:
            impact_score = 0.3  # Low risk
        else:
            impact_score = 0.0  # No impact
    elif feature_name == 'Short_Breath':
        if original_val >= 3:
            impact_score = 0.4  # Strong risk
        elif original_val >= 2:
            impact_score = 0.3  # Moderate risk
        elif original_val >= 1:
            impact_score = 0.2  # Low risk
        else:
            impact_score = 0.0  # No impact
    elif feature_name == 'Exercise':
        impact_score = 0.3 if original_val == 0 else -0.2  # No exercise risk
    elif feature_name == 'Fatty_Food':
        if original_val >= 4:
            impact_score = 0.4  # Strong risk
        elif original_val >= 3:
            impact_score = 0.3  # Moderate risk
        elif original_val >= 2:
            impact_score = 0.2  # Low risk
        else:
            impact_score = 0.0  # No impact
    elif feature_name == 'Stress':
        if original_val >= 4:
            impact_score = 0.4  # Strong risk
        elif original_val >= 3:
            impact_score = 0.3  # Moderate risk
        elif original_val >= 2:
            impact_score = 0.2  # Low risk
        else:
            impact_score = 0.0  # No impact
    else:
        # For other features, use reduced model coefficient
        impact_score = ((original_val - feature_means[i]) / feature_scales[i]) * coefficients[i] * 0.5
    
    # Calculate relative importance
    if total_log_odds != 0:
        relative_importance = abs(impact_score) / abs(total_log_odds) * 100
    else:
        relative_importance = 0.0
    
    # Normalize impact
    normalized_impact = 2 * (1 / (1 + np.exp(-impact_score))) - 1
    
    # Determine impact category
    if impact_score > 0.3:
        category = "Strongly Increases Risk"
    elif impact_score > 0.1:
        category = "Moderately Increases Risk"
    elif impact_score < -0.3:
        category = "Strongly Decreases Risk"
    elif impact_score < -0.1:
        category = "Moderately Decreases Risk"
    else:
        category = "Minimal Impact"
    
    print(f"{feature_name}: {original_val} -> {impact_score:.3f} -> {category}") 