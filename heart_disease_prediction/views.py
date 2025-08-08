import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
import joblib
from django.http import JsonResponse

def calculate_risk_score(row):
    base_score = 0
    
    # Age-based risk
    if row['age'] < 30:
        base_score += 0
    elif row['age'] < 40:
        base_score += 1
    else:
        base_score += 2
        
    # Blood pressure risk
    if row['trestbps'] < 120:  # Normal
        base_score += 0
    elif row['trestbps'] < 130:  # Elevated
        base_score += 1
    elif row['trestbps'] < 140:  # Stage 1
        base_score += 2
    else:  # Stage 2
        base_score += 3
        
    # Cholesterol risk
    if row['chol'] < 200:  # Desirable
        base_score += 0
    elif row['chol'] < 240:  # Borderline high
        base_score += 1
    else:  # High
        base_score += 2
        
    # Other factors
    if row['fbs'] == 1:  # High fasting blood sugar
        base_score += 2
    if row['exang'] == 1:  # Exercise-induced angina
        base_score += 2
    if row['oldpeak'] > 2:  # ST depression
        base_score += 2
        
    return base_score

def train_normal_user_model():
    # Load the dataset
    df = pd.read_csv('heart_disease_prediction/static/dataset/heart.csv')
    
    # Create a copy of the dataset for normal users
    df_normal = df.copy()
    
    # Add risk score column
    df_normal['risk_score'] = df_normal.apply(calculate_risk_score, axis=1)
    
    # Define features and target
    X = df_normal.drop(['target', 'risk_score'], axis=1)
    y = df_normal['target']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train the model with parameters optimized for normal users
    normal_user_model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=3,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    
    # Fit the model
    normal_user_model.fit(X_train_scaled, y_train)
    
    # Save the model and scaler
    joblib.dump(normal_user_model, 'heart_disease_prediction/static/dataset/normal_user_model.joblib')
    joblib.dump(scaler, 'heart_disease_prediction/static/dataset/normal_user_scaler.joblib')
    
    return normal_user_model, scaler

def predict_heart_disease_patient(request):
    if request.method == 'POST':
        try:
            # Load the normal user model and scaler
            normal_user_model = joblib.load('heart_disease_prediction/static/dataset/normal_user_model.joblib')
            scaler = joblib.load('heart_disease_prediction/static/dataset/normal_user_scaler.joblib')
            
            # Get form data
            list_data = []
            age = float(request.POST.get('age'))
            list_data.append(age)
            
            sex = request.POST.get('sex')
            list_data.append(float(1 if sex == "1" else 0))
            
            chest_pain = int(request.POST.get('chest_pain', 0))
            cp_value = 4  # Default to asymptomatic
            if chest_pain == 3:  # Often
                cp_value = 2  # Atypical angina
            elif chest_pain == 2:  # Sometimes
                cp_value = 3  # Non-anginal pain
            list_data.append(float(cp_value))
            
            has_hbp = int(request.POST.get('hypertension', 0))
            trestbps = 130 if has_hbp == 1 else 110
            list_data.append(float(trestbps))
            
            has_cholesterol = int(request.POST.get('high_cholesterol', 0))
            chol = 220 if has_cholesterol == 1 else 170
            list_data.append(float(chol))
            
            has_diabetes = int(request.POST.get('diabetes', 0))
            list_data.append(float(1 if has_diabetes == 1 else 0))
            
            list_data.append(float(0))  # restecg
            
            max_heart_rate = 220 - age
            activity_level = int(request.POST.get('physical_activity', 0))
            if activity_level == 0:  # Never
                max_heart_rate *= 0.85
            elif activity_level == 1:  # Rarely
                max_heart_rate *= 0.9
            elif activity_level == 2:  # Sometimes
                max_heart_rate *= 0.95
            list_data.append(float(max_heart_rate))
            
            shortness_breath = int(request.POST.get('shortness_of_breath', 0))
            list_data.append(float(1 if shortness_breath >= 2 else 0))
            
            stress_level = float(request.POST.get('stress_level', 0))
            oldpeak = stress_level * 0.3
            list_data.append(float(oldpeak))
            
            slope = 2  # Default to flat
            if activity_level >= 2:  # Regular exercise
                slope = 1  # Upsloping
            list_data.append(float(slope))
            
            family_history = int(request.POST.get('family_history', 0))
            list_data.append(float(1 if family_history == 1 else 0))
            
            list_data.append(float(1))  # thal
            
            # Calculate risk score
            input_risk_score = calculate_risk_score(pd.Series({
                'age': age,
                'trestbps': trestbps,
                'chol': chol,
                'fbs': 1 if has_diabetes == 1 else 0,
                'exang': 1 if shortness_breath >= 2 else 0,
                'oldpeak': oldpeak
            }))
            
            # Add risk score to input data
            list_data.append(input_risk_score)
            
            # Convert to numpy array and reshape
            input_data = np.array(list_data).reshape(1, -1)
            
            # Scale the input data
            input_scaled = scaler.transform(input_data)
            
            # Make prediction
            pred = normal_user_model.predict(input_scaled)
            pred_proba = normal_user_model.predict_proba(input_scaled)[0]
            
            # Special handling for young patients
            if age < 35:
                # If young and all vital signs are normal, override to healthy (0)
                if (trestbps < 130 and    # Normal blood pressure
                    chol < 200 and        # Normal cholesterol
                    has_diabetes == 0 and  # Normal blood sugar
                    shortness_breath < 2 and  # No exercise-induced angina
                    oldpeak < 1.0 and     # Normal ST depression
                    input_risk_score < 3): # Low risk score
                    pred = [0]
                    
                # If borderline case (probability close to threshold), lean towards negative
                elif pred_proba[1] < 0.7:  # If less than 70% confident of disease
                    pred = [0]
            
            if pred[0] == 0:
                pred = "<span style='color:green'>You are healthy</span>"
            else:
                pred = "<span style='color:red'>You may have a risk of heart disease</span>"
            
            return JsonResponse({'prediction': pred})
            
        except Exception as e:
            return JsonResponse({'error': str(e)})
    
    return JsonResponse({'error': 'Invalid request method'}) 