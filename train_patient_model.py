import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import json
import datetime
import os

def train_patient_model():
    print("Loading medical_dataset.csv...")
    # Adjust the path to your actual medical_dataset.csv location
    # Assuming it's in the 'media' folder as per your views.py
    csv_path = 'media/medical_dataset.csv' 
    if not os.path.exists(csv_path):
        print(f"Error: Dataset not found at {csv_path}. Please ensure the file exists.")
        return None, None, 0.0

    df = pd.read_csv(csv_path)

    # Ensure 'Gender' column is mapped if not already numerical
    if 'Gender' in df.columns and df['Gender'].dtype == 'object':
        df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
        print("Mapped 'Gender' column to numerical values.")

    # Define features based on your patient_input_data_for_prediction in views.py
    features = [
        'Age', 'Gender', 'Height', 'Weight', 'BMI', 'Smoke', 'Time_of_Smoking', 'Frequency_of_smoking',
        'High_Blood_Pressure', 'Diabetes', 'High_Cholesterol', 'Family_History',
        'Chest_Pain', 'Chest_Pain_Severity', 'Short_Breath', 'Short_Breath_Duration',
        'Exercise', 'Fatty_Food', 'Stress'
    ]
    
    # Ensure all features exist in the dataframe before dropping 'Result'
    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        print(f"Error: Missing expected features in dataset: {missing_features}")
        return None, None, 0.0

    X = df[features]
    y = df['Result'] # Assuming 'Result' is your target column

    print("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Training LogisticRegression (max_iter=1000)...")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)

    print("Evaluating model...")
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    # Print detailed metrics to backend console
    print("=" * 60)
    print("PATIENT MODEL EVALUATION METRICS (STANDALONE TRAINING)")
    print("=" * 60)
    print(f"Model Accuracy: {acc:.4f} ({acc*100:.2f}%)")
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
    from sklearn.metrics import f1_score
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

    print("Saving model and scaler...")
    # Save model and scaler in a dictionary as per your views.py loading logic
    joblib.dump({'model': model, 'scaler': scaler}, 'patient_model.pkl')
    
    metrics = {
        'accuracy': acc,
        'feature_names': features, # Save the feature names
        'confusion_matrix': cm.tolist(),
        'classification_report': report,
        'trained_on': str(datetime.datetime.now()),
        'model_type': 'LogisticRegression'
    }
    with open('patient_model_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4) # Use indent for readability

    # This file is now redundant as metrics.json stores accuracy
    # with open('patient_model_acc.txt', 'w') as f:
    #     f.write(str(acc))
    
    print("Done! Model and scaler have been saved.")
    return model, scaler, acc

if __name__ == "__main__":
    train_patient_model()