from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.shortcuts import render, redirect
import datetime
import os
import joblib
import pandas as pd
import json

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

from .forms import DoctorForm
from .models import *
from django.contrib.auth import authenticate, login, logout
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from django.http import HttpResponse, JsonResponse
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score

from datetime import datetime, date

def print_model_metrics(y_test, y_pred, model_name="Model", accuracy=None):
    """
    Utility function to print detailed model evaluation metrics to backend console
    """
    if accuracy is None:
        accuracy = accuracy_score(y_test, y_pred)
    
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    print("=" * 60)
    print(f"{model_name.upper()} EVALUATION METRICS")
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
    
    return {
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'classification_report': report,
        'f1_scores': {
            'healthy': f1_class_0,
            'unhealthy': f1_class_1,
            'macro': f1_macro,
            'weighted': f1_weighted
        }
    }

# Create your views here.

def Home(request):
    # If user is not authenticated, show the landing page
    if not request.user.is_authenticated:
        return render(request, 'home.html')

    # For authenticated users, redirect based on their role
    if request.user.is_staff:
        return redirect('admin_home')
    elif hasattr(request.user, 'patient'):
        return redirect('patient_home')
    elif hasattr(request.user, 'doctor'):
        return redirect('doctor_home')
    
    # If the user is authenticated but not staff, patient, or doctor, redirect to login
    return redirect('login')

@login_required(login_url="login")
def Admin_Home(request):
    dis = Search_Data.objects.all()
    pat = Patient.objects.all()
    doc = Doctor.objects.all()
    feed = Feedback.objects.all()
    unread_feedback_count = Feedback.objects.filter(is_read=False).count()

    d = {'dis':dis.count(),'pat':pat.count(),'doc':doc.count(),'feed':feed.count(), 'unread_feedback_count': unread_feedback_count}
    return render(request,'admin_home.html',d)

@login_required(login_url="login")
def assign_status(request,pid):
    doctor = Doctor.objects.get(id=pid)
    if doctor.status == 1:
        doctor.status = 2
        messages.success(request, 'Selected doctor are successfully withdraw his approval.')
    else:
        doctor.status = 1
        messages.success(request, 'Selected doctor are successfully approved.')
    doctor.save()
    return redirect('view_doctor')

@login_required(login_url="login")
def User_Home(request):
    user = request.user
    try:
        patient_profile = user.patient  # Assuming a OneToOneField from User to Patient
    except: # Fallback in case of Doctor, though patient_home should only be for patients
        patient_profile = None
    
    context = {
        'pro': patient_profile
    }
    return render(request,'patient_home.html', context)

@login_required(login_url="login")
def Doctor_Home(request):
    return render(request,'doctor_home.html')

def About(request):
    return render(request,'about.html')

def Contact(request):
    # Prepare user data for auto-filling (for logged-in users)
    user_data = {}
    if request.user.is_authenticated:
        user_data = {
            'name': f"{request.user.first_name} {request.user.last_name}".strip(),
            'email': request.user.email,
        }
    
    if request.method == "POST":
        try:
            name = request.POST.get('Name', '').strip()
            email = request.POST.get('Email', '').strip()
            subject = request.POST.get('Subject', '').strip()
            message = request.POST.get('Message', '').strip()
            contact = None # Initialize contact to None

            # Validate required fields
            if not name or not email or not subject or not message:
                messages.error(request, 'Please fill in all required fields.')
                return redirect('contact')

            # Basic email validation
            if '@' not in email or '.' not in email:
                messages.error(request, 'Please enter a valid email address.')
                return redirect('contact')

            # If the user is authenticated, try to get their contact number
            if request.user.is_authenticated:
                if hasattr(request.user, 'patient') and request.user.patient.contact:
                    contact = request.user.patient.contact
                elif hasattr(request.user, 'doctor') and request.user.doctor.contact:
                    contact = request.user.doctor.contact

            # Create a new Feedback object
            Feedback.objects.create(
                name=name,
                email=email,
                contact=contact, # Pass the retrieved contact
                subject=subject,
                messages=message
            )
            messages.success(request, 'Your message has been sent successfully! We will get back to you soon.')
            return redirect('contact') # Redirect back to the contact page after submission
            
        except Exception as e:
            print(f"Error in contact form: {str(e)}")
            messages.error(request, 'Sorry, there was an error sending your message. Please try again.')
            return redirect('contact')

    return render(request, 'contact.html', {'user_data': user_data})


def Gallery(request):
    return render(request,'gallery.html')


def Login_User(request):
    error = ""
    if request.method == "POST":
        form_type = request.POST.get('form_type')
        
        if form_type == "login":  # Login request
            u = request.POST['uname']
            p = request.POST['pwd']
            user = authenticate(username=u, password=p)
            sign = ""
            if user:
                try:
                    sign = Patient.objects.get(user=user)
                except:
                    pass
                if sign:
                    login(request, user)
                    return redirect('patient_home')
                else:
                    pure=False
                    try:
                        pure = Doctor.objects.get(status=1,user=user)
                    except:
                        pass
                    if pure:
                        login(request, user)
                        return redirect('doctor_home')
                    else:
                        login(request, user)
                        error="notmember"
            else:
                error="not"
        elif form_type == "register":  # Registration request
            try:
                f = request.POST['fname']
                l = request.POST['lname']
                u = request.POST['uname']
                e = request.POST['email']
                p = request.POST['pwd']
                d = request.POST['dob']
                con = request.POST['contact']
                add = request.POST['add']
                type = request.POST['type']
                im = request.FILES['image']
                
                # Validate contact number
                if not con.isdigit() or len(con) != 10:
                    error = "invalid_contact"
                    return render(request, 'login.html', {'error': error, 'show_signup': True})
                
                # Check if username already exists
                if User.objects.filter(username=u).exists():
                    error = "username_exists"
                    return render(request, 'login.html', {'error': error, 'show_signup': True})
                else:
                    user = User.objects.create_user(email=e, username=u, password=p, first_name=f, last_name=l)
                    if type == "Patient":
                        Patient.objects.create(user=user, contact=con, address=add, image=im, dob=d)
                    else:
                        Doctor.objects.create(dob=d, image=im, user=user, contact=con, address=add, status=2)
                    error = "create"
                    return render(request, 'login.html', {'error': error, 'show_signup': True})
            except Exception as e:
                error = "registration_error"
                print(f"Registration error: {str(e)}")
                return render(request, 'login.html', {'error': error, 'show_signup': True})
    
    d = {'error': error}
    return render(request, 'login.html', d)

def Login_admin(request):
    error = ""
    if request.method == "POST":
        u = request.POST['uname']
        p = request.POST['pwd']
        user = authenticate(username=u, password=p)
        if user is not None:
            if user.is_staff:
                login(request, user)
                return redirect('admin_home')
            else:
                error = "not"  # Not an admin user
        else:
            error = "invalid"  # Invalid credentials
    d = {'error': error}
    return render(request, 'admin_login.html', d)

def Signup_User(request):
    error = ""
    if request.method == 'POST':
        f = request.POST['fname']
        l = request.POST['lname']
        u = request.POST['uname']
        e = request.POST['email']
        p = request.POST['pwd']
        d = request.POST['dob']
        con = request.POST['contact']
        add = request.POST['add']
        type = request.POST['type']
        im = request.FILES['image']
        dat = datetime.date.today()
        user = User.objects.create_user(email=e, username=u, password=p, first_name=f,last_name=l)
        if type == "Patient":
            Patient.objects.create(user=user,contact=con,address=add,image=im,dob=d)
        else:
            Doctor.objects.create(dob=d,image=im,user=user,contact=con,address=add,status=2)
        error = "create"
    d = {'error':error}
    return render(request,'register.html',d)

def Logout(request):
    logout(request)
    return redirect('home')

@login_required(login_url="login")
def Change_Password(request):
    sign = 0
    user = User.objects.get(username=request.user.username)
    error = ""
    if not request.user.is_staff:
        try:
            sign = Patient.objects.get(user=user)
            if sign:
                error = "pat"
        except:
            sign = Doctor.objects.get(user=user)
    terror = ""
    if request.method=="POST":
        n = request.POST['pwd1']
        c = request.POST['pwd2']
        o = request.POST['pwd3']
        if c == n:
            u = User.objects.get(username__exact=request.user.username)
            u.set_password(n)
            u.save()
            terror = "yes"
        else:
            terror = "not"
    d = {'error':error,'terror':terror,'data':sign}
    return render(request,'change_password.html',d)


def preprocess_inputs(df, scaler):
    df = df.copy()
    # Split df into X and y
    y = df['target'].copy()
    X = df.drop('target', axis=1).copy()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    return X, y


def retrain_heart_model():
    try:
        csv_path = 'media/heart.csv'
        model_path = 'heart_model.pkl'
        acc_path = 'heart_model_acc.txt'
        scaler_path = 'heart_model_scaler.pkl'
        
        # Load and preprocess data
        df = pd.read_csv(csv_path)
        print("Data loaded, shape:", df.shape)
        
        # Basic data validation
        required_columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                          'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
        
        if not all(col in df.columns for col in required_columns):
            raise ValueError("Missing required columns in the dataset")
        
        print("Class distribution:")
        print(df['target'].value_counts(normalize=True) * 100)
        
        X = df[['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
               'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']]
        y = df['target']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        print("Data split - train:", X_train.shape, "test:", X_test.shape)

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        print("Data scaled")

        # Train a Logistic Regression model with balanced class weights
        model = LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            random_state=42
        )
        print("Training Logistic Regression model...")
        model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        
        # Print detailed metrics to backend console
        print("=" * 60)
        print("HEART DISEASE MODEL EVALUATION METRICS")
        print("=" * 60)
        print(f"Model Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print("\nCONFUSION MATRIX:")
        print("                 Predicted")
        print("                 No Disease | Disease")
        print(f"Actual No Disease |    {cm[0][0]:4d}    |   {cm[0][1]:4d}")
        print(f"Actual Disease    |    {cm[1][0]:4d}    |   {cm[1][1]:4d}")
        print(f"\nTrue Negatives (TN): {cm[0][0]} - Correctly predicted healthy")
        print(f"False Positives (FP): {cm[0][1]} - Healthy predicted as unhealthy")
        print(f"False Negatives (FN): {cm[1][0]} - Unhealthy predicted as healthy")
        print(f"True Positives (TP): {cm[1][1]} - Correctly predicted unhealthy")
        print("\nCLASSIFICATION REPORT:")
        print(report)
        
        # Calculate and print F1 scores for each class
        from sklearn.metrics import f1_score
        f1_class_0 = f1_score(y_test, y_pred, pos_label=0)  # F1 for healthy class
        f1_class_1 = f1_score(y_test, y_pred, pos_label=1)  # F1 for disease class
        f1_macro = f1_score(y_test, y_pred, average='macro')  # Macro average F1
        f1_weighted = f1_score(y_test, y_pred, average='weighted')  # Weighted average F1
        
        print("\nF1 SCORES:")
        print(f"F1 Score (Healthy Class 0): {f1_class_0:.4f}")
        print(f"F1 Score (Disease Class 1): {f1_class_1:.4f}")
        print(f"F1 Score (Macro Average): {f1_macro:.4f}")
        print(f"F1 Score (Weighted Average): {f1_weighted:.4f}")
        print("=" * 60)
        
        # Save model and scaler
        print("Saving model and scaler...")
        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)
        
        # Save metrics in a more comprehensive format
        metrics = {
            'accuracy': accuracy,
            'confusion_matrix': cm.tolist(),
            'classification_report': report,
            'feature_names': X.columns.tolist(),
            'trained_on': str(datetime.now()),
            'model_type': 'LogisticRegression'
        }
        with open(acc_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        print("Model training completed successfully!")
        return model, scaler, accuracy
        
    except Exception as e:
        print(f"Error in model training: {str(e)}")
        raise

def prdict_heart_disease(list_data):
    try:
        model_path = 'heart_model.pkl'
        scaler_path = 'heart_model_scaler.pkl'
        acc_path = 'heart_model_acc.txt'
        
        # Check if model files exist and are valid
        if not all(os.path.exists(p) for p in [model_path, scaler_path, acc_path]):
            print("Model files not found. Retraining model...")
            model, scaler, accuracy = retrain_heart_model()
        else:
            # Load existing model and scaler
            try:
                model = joblib.load(model_path)
                scaler = joblib.load(scaler_path)
                with open(acc_path, 'r') as f:
                    content = f.read()
                    try:
                        # Try to parse as JSON (new format)
                        metrics = json.loads(content)
                        accuracy = float(metrics.get('accuracy', 0.0)) * 100  # Correctly get 'accuracy' and convert to percentage
                    except json.JSONDecodeError:
                        # Fallback to old float format if JSON decoding fails
                        try:
                            accuracy = float(content)
                            if accuracy <= 1.0:
                                accuracy = accuracy * 100
                        except ValueError:
                            accuracy = 0.0  # Default or error handling
            except Exception as e:
                print(f"Error loading model: {str(e)}. Retraining...")
                model, scaler, accuracy = retrain_heart_model()
        
        # Ensure input is in correct format
        if not isinstance(list_data, (list, np.ndarray)):
            raise ValueError("Input must be a list or numpy array")
            
        if len(list_data) != 13:
            raise ValueError(f"Expected 13 features, got {len(list_data)}")
        
        # Convert to numpy array and reshape
        X = np.array(list_data).reshape(1, -1)
        
        # Print input data for debugging
        print("Input data:", list_data)
        
        # Scale features
        X_scaled = scaler.transform(X)
        print("Scaled input:", X_scaled)
        
        # Make prediction
        pred = model.predict(X_scaled)
        pred_proba = model.predict_proba(X_scaled)[0]
        print("Prediction:", pred[0])
        print("Prediction probabilities:", pred_proba)
        
        # Print current model metrics for reference
        print("\n" + "=" * 50)
        print("CURRENT HEART DISEASE MODEL METRICS")
        print("=" * 50)
        print(f"Model Accuracy: {accuracy:.2f}%")
        print("Model Type: Logistic Regression")
        print("Features Used: 13 medical parameters")
        
        # Load and display saved metrics if available
        try:
            with open(acc_path, 'r') as f:
                content = f.read()
                try:
                    metrics = json.loads(content)
                    if 'confusion_matrix' in metrics:
                        cm_saved = metrics['confusion_matrix']
                        print(f"Saved Confusion Matrix: TN={cm_saved[0][0]}, FP={cm_saved[0][1]}, FN={cm_saved[1][0]}, TP={cm_saved[1][1]}")
                        
                        # Calculate and display F1 scores from confusion matrix
                        tn, fp, fn, tp = cm_saved[0][0], cm_saved[0][1], cm_saved[1][0], cm_saved[1][1]
                        
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
                        
                        print("\nF1 SCORES (Calculated from Saved Confusion Matrix):")
                        print(f"F1 Score (Healthy Class 0): {f1_0:.4f}")
                        print(f"F1 Score (Disease Class 1): {f1_1:.4f}")
                        print(f"F1 Score (Macro Average): {f1_macro:.4f}")
                        print(f"F1 Score (Weighted Average): {f1_weighted:.4f}")
                        
                    if 'classification_report' in metrics:
                        print("\nCLASSIFICATION REPORT:")
                        print(metrics['classification_report'])
                    if 'f1_scores' in metrics:
                        print("\nSAVED F1 SCORES:")
                        f1_scores = metrics['f1_scores']
                        for key, value in f1_scores.items():
                            print(f"F1 Score ({key}): {value:.4f}")
                except json.JSONDecodeError:
                    pass
        except Exception as e:
            print(f"Could not load saved metrics: {e}")
        print("=" * 50)
        
        # Calculate feature importance/impact for explainability
        feature_impacts = []
        if hasattr(model, 'coef_') and model.coef_.shape[0] == 1:
            # For binary classification with a single output (Logistic Regression)
            coefficients = model.coef_[0]  # Get the coefficients for the positive class (1=unhealthy)
            
            # Get the mean and scale from the scaler for each feature
            feature_means = scaler.mean_
            feature_scales = scaler.scale_
            
            # Define feature names for the doctor model (13 features)
            feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
            
            for i, feature_name in enumerate(feature_names):
                # Original value from list_data for context
                original_val = list_data[i] if i < len(list_data) else 0
                
                # Calculate the impact using the original value and coefficient
                # Impact = (original_value - mean) / scale * coefficient
                # This gives us the contribution of this feature to the log-odds
                impact_score = ((original_val - feature_means[i]) / feature_scales[i]) * coefficients[i]
                
                # Normalize the impact to be between -1 and 1 for better interpretability
                # We use the feature's scale to normalize the impact
                # Ensure denominator is not zero
                denominator = abs(coefficients[i]) * feature_scales[i]
                if denominator == 0:
                    normalized_impact = 0.0
                else:
                    normalized_impact = impact_score / denominator
                
                # Calculate the relative importance (percentage contribution)
                # Sum of absolute impact scores for normalization
                total_abs_impact = sum(abs(((list_data[j] - feature_means[j]) / feature_scales[j]) * coefficients[j]) for j in range(len(feature_names)) if j < len(list_data))
                
                if total_abs_impact == 0:
                    relative_importance = 0.0
                else:
                    relative_importance = abs(impact_score) / total_abs_impact
                
                feature_impacts.append({
                    'feature': feature_name,
                    'value': original_val,
                    'coefficient': round(coefficients[i], 3),
                    'impact': round(impact_score, 3),
                    'normalized_impact': round(normalized_impact, 3),
                    'relative_importance': round(relative_importance * 100, 1)  # Convert to percentage
                })
        
        return accuracy, pred[0], pred_proba, feature_impacts
        
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        raise

@login_required(login_url="login")
def add_doctor(request,pid=None):
    doctor = None
    if pid:
        doctor = Doctor.objects.get(id=pid)
    if request.method == "POST":
        form = DoctorForm(request.POST, request.FILES, instance = doctor)
        if form.is_valid():
            new_doc = form.save()
            new_doc.status = 1
            if not pid:
                user = User.objects.create_user(password=request.POST['password'], username=request.POST['username'], first_name=request.POST['first_name'], last_name=request.POST['last_name'])
                new_doc.user = user
            new_doc.save()
            return redirect('view_doctor')
    d = {"doctor": doctor}
    return render(request, 'add_doctor.html', d)

def calculate_risk_score(row):
    score = 0

    if row['age'] > 50:
        score += 2
    elif row['age'] > 40:
        score += 1

    if row['chol'] > 240:
        score += 2
    elif row['chol'] > 200:
        score += 1

    if row['trestbps'] > 140:
        score += 2
    elif row['trestbps'] > 120:
        score += 1

    if row['thalach'] < 100:
        score += 2
    elif row['thalach'] < 120:
        score += 1

    if row['fbs'] == 1:
        score += 1

    if row['exang'] == 1:
        score += 1

    if row['oldpeak'] > 2.0:
        score += 2
    elif row['oldpeak'] > 1.0:
        score += 1

    if row['ca'] > 0:
        score += row['ca']

    if row['thal'] in [2, 3]:
        score += 1

    return score


@login_required(login_url="login")
def add_heartdetail(request):
    if request.method == "POST":
        # Get patient name and contact from the form
        patient_name = request.POST.get('patient_name', '')
        patient_contact = request.POST.get('patient_contact', '')

        # Only extract the fields your model expects, in the correct order
        fields = [
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
            'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
        ]
        list_data = []
        missing_fields = []
        for field in fields:
            value = request.POST.get(field)
            if value is None or value == '':
                missing_fields.append(field)
                continue
            
            # Data cleaning/conversion for model features
            if field == 'sex':
                if str(value).lower() in ['1', 'male', 'm']:
                    value = 1
                else:
                    value = 0
            elif field == 'cp':
                value = int(value)  # Should be 1,2,3,4
            elif field == 'fbs':
                value = 1 if str(value).lower() in ['1', 'true', 'yes'] else 0
            elif field == 'restecg':
                value = int(value)  # 0,1,2
            elif field == 'exang':
                value = 1 if str(value).lower() in ['1', 'true', 'yes'] else 0
            elif field == 'slope':
                value = int(value)  # 1,2,3
            elif field == 'ca':
                value = int(value)  # 0,1,2,3,4
            elif field == 'thal':
                value = int(value)  # 1,2,3
            
            try:
                list_data.append(float(value))
            except ValueError:
                missing_fields.append(field)

        if missing_fields:
            error_msg = f"Missing or invalid input for: {', '.join(missing_fields)}"
            if request.headers.get('x-requested-with') == 'XMLHttpRequest':
                return JsonResponse({'error': error_msg}, status=400)
            messages.error(request, error_msg)
            return render(request, 'add_heartdetail.html')

        # Create a dictionary to save all input data including name and contact
        input_data_to_save = {
            'patient_name': patient_name,
            'patient_contact': patient_contact,
            'features': dict(zip(fields, list_data)) # Save features with their names
        }

        # Load model and scaler if available
        model_path = 'heart_model.pkl'
        acc_path = 'heart_model_acc.txt'
        scaler_path = 'heart_model_scaler.pkl'
        if not os.path.exists(model_path) or not os.path.exists(acc_path):
            retrain_heart_model()
        model = joblib.load(model_path)
        with open(acc_path, 'r') as f:
            content = f.read()
            try:
                # Try to parse as JSON (new format)
                metrics = json.loads(content)
                accuracy = float(metrics.get('accuracy', 0.0)) * 100  # Correctly get 'accuracy' and convert to percentage
            except json.JSONDecodeError:
                # Fallback to old float format if JSON decoding fails
                try:
                    accuracy = float(content)
                    if accuracy <= 1.0:
                        accuracy = accuracy * 100
                except ValueError:
                    accuracy = 0.0  # Default or error handling
        # If you have a scaler, load and apply it
        # if os.path.exists(scaler_path):
        #     scaler = joblib.load(scaler_path)
        #     list_data_scaled = scaler.transform([list_data])
        # else:
        #     list_data_scaled = [list_data]
        print("Doctor model input features:", list_data)
        accuracy, pred, pred_proba, feature_impacts = prdict_heart_disease(list_data)

        # Get prediction probability
        healthy_prob = pred_proba[0]  # Probability of being healthy (class 0)
        unhealthy_prob = pred_proba[1]  # Probability of being unhealthy (class 1)
        
        # Use a threshold of 0.5 for classification
        pred_value = 1 if unhealthy_prob > 0.5 else 0

        # Save the search data (only use valid fields for Search_Data)
        search_data = None
        patient = None
        doctor = None
        try:
            patient = Patient.objects.get(user=request.user)
        except Patient.DoesNotExist:
            try:
                doctor = Doctor.objects.get(user=request.user)
            except Doctor.DoesNotExist:
                pass

        if patient or doctor:
            search_data = Search_Data.objects.create(
                patient=patient,
                doctor=doctor,
                prediction_accuracy=accuracy, 
                result=pred_value,
                values_list=json.dumps(input_data_to_save)
            )
        # Create history entry
        if search_data:
            PredictionHistory.objects.create(
                search_data=search_data,
                prediction_accuracy=accuracy,
                result=pred_value,
                values_list=json.dumps(input_data_to_save)
            )

        print(f"Doctor Final prediction: {pred_value} (0=healthy, 1=unhealthy)")
        print(f"Doctor Confidence: healthy={healthy_prob:.1f}%, unhealthy={unhealthy_prob:.1f}%")
        
        # Return JSON if AJAX request
        if request.headers.get('x-requested-with') == 'XMLHttpRequest':
            return JsonResponse({
                'prediction': int(pred_value),
                'accuracy': float(accuracy),
                'healthy_prob': float(healthy_prob),
                'unhealthy_prob': float(unhealthy_prob),
            })
        # Instead of redirecting, render the template directly
        pred_text = "<span class='healthy'>Healthy</span>" if int(pred_value) == 0 else "<span class='unhealthy'>Unhealthy</span>"
        
        # Retrieve patient or doctor for context if needed in the template
        patient = None
        doctor = None
        try:
            patient = Patient.objects.get(user=request.user)
        except Patient.DoesNotExist:
            try:
                doctor = Doctor.objects.get(user=request.user)
            except Doctor.DoesNotExist:
                pass
        
        if patient:
            # Get patient's city
            patient_city = extract_city(patient.address)
            
            # Find doctors in the same city who are authorized (status=1)
            doctors = Doctor.objects.filter(
                status=1,  # Only authorized doctors
                address__icontains=patient_city  # Match by city
            ).order_by('category')  # Order by specialization
            
            # If no doctors found in exact city, try nearby cities (you can expand this logic)
            if not doctors.exists() and patient_city:
                # Try to find doctors in nearby cities (this is a simple example)
                # You could expand this with a proper city database and distance calculation
                doctors = Doctor.objects.filter(
                    status=1,
                    address__icontains=patient_city.split()[0]  # Try matching first word of city
                ).order_by('category')
            
            return render(request, 'predict_disease.html', {
                'pred': pred_text, 
                'accuracy': accuracy, 
                'doctor': doctors,
                'patient_city': patient_city,
                'feature_impacts': feature_impacts
            })
        elif doctor:
            return render(request, 'predict_disease.html', {'pred': pred_text, 'accuracy': accuracy, 'feature_impacts': feature_impacts})
        else:
            return render(request, 'predict_disease.html', {'pred': pred_text, 'accuracy': accuracy, 'feature_impacts': feature_impacts})
    return render(request, 'add_heartdetail.html')

@login_required(login_url="login")
def predict_desease(request, pred, accuracy):
    try:
        pred_val = int(float(pred))
        if pred_val == 0:
            pred_text = "<span class='healthy'>Healthy</span>"
        else:
            pred_text = "<span class='unhealthy'>Unhealthy</span>"
        if request.user.patient_set.all().exists():
            patient = Patient.objects.get(user=request.user)
            doctor = Doctor.objects.filter(address__icontains=patient.address)
            return render(request, 'predict_disease.html', {'pred': pred_text, 'accuracy':accuracy, 'doctor':doctor})
        elif request.user.doctor_set.all().exists():
            return render(request, 'predict_disease.html', {'pred': pred_text, 'accuracy':accuracy})
    except Exception as e:
        print("Prediction error:", e)
        return redirect('home')
    return render(request, 'predict_disease.html', {'pred': pred, 'accuracy':accuracy})

@login_required(login_url="login")
def view_search_pat(request):
    try:
        # Fetch data based on user role
        if request.user.is_staff:
            data = Search_Data.objects.all().order_by('-created')
        elif hasattr(request.user, 'doctor'):
            doctor = request.user.doctor
            data = Search_Data.objects.filter(doctor=doctor).order_by('-created')
        elif hasattr(request.user, 'patient'):
            patient = request.user.patient
            data = Search_Data.objects.filter(patient=patient).order_by('-created')
        else:
            data = []

        # Process each item to add parsed_values and contact info
        for item in data:
            # Initialize parsed_values
            item.parsed_values = {}
            if item.values_list:
                try:
                    # Attempt to load values_list as JSON
                    item.parsed_values = json.loads(item.values_list)
                except json.JSONDecodeError:
                    print(f"[DEBUG] Error decoding JSON for Search_Data ID {item.id}: {item.values_list}")
                    item.parsed_values = {} # Set to empty dict on error

            # Determine patient name and contact for display
            if item.patient:
                # If linked to a registered patient, use their name and contact
                item.patient_name_from_values = f"{item.patient.user.first_name} {item.patient.user.last_name}".strip()
                item.patient_contact_from_values = item.patient.contact
            else:
                # Otherwise, extract from parsed_values (for doctor mode entries)
                item.patient_name_from_values = item.parsed_values.get('patient_name', 'N/A')
                item.patient_contact_from_values = item.parsed_values.get('patient_contact', 'N/A')

        # Determine if the current user is a doctor or staff for template logic
        is_doctor_or_staff = hasattr(request.user, 'doctor') or request.user.is_staff

        return render(request, 'view_search_pat.html', {'data': data, 'is_doctor_or_staff': is_doctor_or_staff})

    except Exception as e:
        print(f"[DEBUG] Error in view_search_pat: {e}")
        # Return an empty list and the flag in case of an error to prevent a blank page
        return render(request, 'view_search_pat.html', {'data': [], 'is_doctor_or_staff': False})

@login_required(login_url="login")
def delete_doctor(request,pid):
    doc = Doctor.objects.get(id=pid)
    doc.delete()
    return redirect('view_doctor')

@login_required(login_url="login")
def delete_feedback(request,pid):
    doc = Feedback.objects.get(id=pid)
    doc.delete()
    return redirect('view_feedback')

@login_required(login_url="login")
def delete_patient(request,pid):
    doc = Patient.objects.get(id=pid)
    doc.delete()
    return redirect('view_patient')

@login_required(login_url="login")
def delete_searched(request,pid):
    doc = Search_Data.objects.get(id=pid)
    doc.delete()
    return redirect('view_search_pat')

@login_required(login_url="login")
def View_Doctor(request):
    doc = Doctor.objects.all()
    d = {'doc':doc}
    return render(request,'view_doctor.html',d)

@login_required(login_url="login")
def View_Patient(request):
    patient = Patient.objects.all()
    d = {'patient':patient}
    return render(request,'view_patient.html',d)

@login_required(login_url="login")
def View_Feedback(request):
    feedback_messages = Feedback.objects.all()
    for message in feedback_messages:
        if not message.is_read:
            message.is_read = True
            message.save()
    d = {'dis':feedback_messages}
    return render(request,'view_feedback.html',d)

@login_required(login_url="login")
def View_My_Detail(request):
    terror = ""
    user = User.objects.get(id=request.user.id)
    error = ""
    try:
        sign = Patient.objects.get(user=user)
        error = "pat"
    except:
        sign = Doctor.objects.get(user=user)
    d = {'error': error,'pro':sign}
    return render(request,'profile_doctor.html',d)

@login_required(login_url="login")
def Edit_Doctor(request,pid):
    doc = Doctor.objects.get(id=pid)
    error = ""
    # type = Type.objects.all()
    if request.method == 'POST':
        f = request.POST['fname']
        l = request.POST['lname']
        e = request.POST['email']
        con = request.POST['contact']
        add = request.POST['add']
        cat = request.POST['type']
        try:
            im = request.FILES['image']
            doc.image=im
            doc.save()
        except:
            pass
        dat = datetime.date.today()
        doc.user.first_name = f
        doc.user.last_name = l
        doc.user.email = e
        doc.contact = con
        doc.category = cat
        doc.address = add
        doc.user.save()
        doc.save()
        error = "create"
    d = {'error':error,'doc':doc,'type':type}
    return render(request,'edit_doctor.html',d)

@login_required(login_url="login")
def Edit_My_deatail(request):
    terror = ""
    print("Hii welcome")
    user = User.objects.get(id=request.user.id)
    error = ""
    # type = Type.objects.all()
    try:
        sign = Patient.objects.get(user=user)
        error = "pat"
    except:
        sign = Doctor.objects.get(user=user)
    if request.method == 'POST':
        f = request.POST['fname']
        l = request.POST['lname']
        e = request.POST['email']
        con = request.POST['contact']
        add = request.POST['add']
        try:
            im = request.FILES['image']
            sign.image = im
            sign.save()
        except:
            pass
        to1 = date.today()
        sign.user.first_name = f
        sign.user.last_name = l
        sign.user.email = e
        sign.contact = con
        if error != "pat":
            cat = request.POST['type']
            sign.category = cat
            sign.save()
        sign.address = add
        sign.user.save()
        sign.save()
        terror = "create"
    d = {'error':error,'terror':terror,'doc':sign}
    return render(request,'edit_profile.html',d)

@login_required(login_url='login')
def sent_feedback(request):
    terror = None
    if request.method == "POST":
        name = request.user.username  # Use the logged-in user's username as the name
        email = request.user.email    # Use the logged-in user's email
        message = request.POST['msg']
        subject = "Feedback from user" # Default subject since there's no input field for it
        contact = None

        print(f"DEBUG: request.user: {request.user}")
        if hasattr(request.user, 'patient'):
            print(f"DEBUG: request.user has patient attribute. patient: {request.user.patient}")
            if request.user.patient.contact:
                contact = request.user.patient.contact
                print(f"DEBUG: Retrieved patient contact: {contact}")
        elif hasattr(request.user, 'doctor'):
            print(f"DEBUG: request.user has doctor attribute. doctor: {request.user.doctor}")
            if request.user.doctor.contact:
                contact = request.user.doctor.contact
                print(f"DEBUG: Retrieved doctor contact: {contact}")
        else:
            print("DEBUG: request.user has neither patient nor doctor attribute.")

        Feedback.objects.create(name=name, email=email, subject=subject, messages=message, contact=contact)
        terror = "create"
    return render(request, 'sent_feedback.html',{'terror':terror})

@login_required(login_url="login")
def view_prediction_history(request, search_id):
    try:
        # Get the main Search_Data object
        search_data = Search_Data.objects.get(id=search_id)

        # Prepare formatted values for the 'Original Prediction' section (search_data)
        main_parsed_data = {}
        if search_data.values_list:
            try:
                main_parsed_data = json.loads(search_data.values_list)
            except json.JSONDecodeError:
                print(f"Error decoding main values_list JSON for search_id {search_id}")
                main_parsed_data = {}

        main_formatted_values = {}
        # Determine the source of features based on the structure of main_parsed_data
        if isinstance(main_parsed_data, dict):
            # If it's a doctor mode entry, 'features' will contain the medical data
            if 'features' in main_parsed_data:
                # Combine patient_name, patient_contact with the 'features' dictionary
                data_source_for_display = {
                    'patient_name': main_parsed_data.get('patient_name'),
                    'patient_contact': main_parsed_data.get('patient_contact'),
                    **main_parsed_data['features']
                }
            else:
                # This is likely from add_heartdetail_patient, which saves a dict of human-readable keys
                data_source_for_display = main_parsed_data
        else:
            # Fallback for unexpected formats (e.g., if it was just a list directly)
            print(f"Unexpected format for main values_list for search_id {search_id}: {type(main_parsed_data)}")
            data_source_for_display = {}


        for key, value in data_source_for_display.items():
            # Clean up key names: replace underscores with spaces and title case
            display_key = key.replace('_', ' ').title()

            # Apply specific mappings for known feature names
            if key == 'sex':
                main_formatted_values['Sex'] = 'Male' if value == 1.0 else 'Female'
            elif key == 'cp':
                cp_map = {1.0: 'Typical Angina', 2.0: 'Atypical Angina', 3.0: 'Non-anginal Pain', 4.0: 'Asymptomatic'}
                main_formatted_values['Chest Pain Type'] = cp_map.get(value, str(value))
            elif key == 'fbs':
                main_formatted_values['Fasting Blood Sugar'] = 'Yes (>120 mg/dl)' if value == 1.0 else 'No (<=120 mg/dl)'
            elif key == 'restecg':
                restecg_map = {0.0: 'Normal', 1.0: 'ST-T wave abnormality', 2.0: 'Left ventricular hypertrophy'}
                main_formatted_values['Resting ECG'] = restecg_map.get(value, str(value))
            elif key == 'exang':
                main_formatted_values['Exercise Induced Angina'] = 'Yes' if value == 1.0 else 'No'
            elif key == 'slope':
                slope_map = {1.0: 'Upsloping', 2.0: 'Flat', 3.0: 'Downsloping'}
                main_formatted_values['Slope of Peak Exercise ST Segment'] = slope_map.get(value, str(value))
            elif key == 'ca':
                main_formatted_values['Number of Major Vessels'] = str(int(value)) if value is not None else 'N/A'
            elif key == 'thal':
                thal_map = {1.0: 'Fixed Defect', 2.0: 'Normal', 3.0: 'Reversible Defect'}
                main_formatted_values['Thalassemia'] = thal_map.get(value, str(value))
            # New patient-specific mappings
            elif key == 'Gender':
                main_formatted_values['Gender'] = 'Male' if value == 1 else 'Female'
            elif key == 'Smoke':
                main_formatted_values['Smoking'] = 'Yes' if value == 1 else 'No'
            elif key == 'High_Blood_Pressure':
                hp_map = {0: 'No', 1: 'Yes', 2: 'Not sure'}
                main_formatted_values['High Blood Pressure'] = hp_map.get(value, str(value))
            elif key == 'Diabetes':
                diabetes_map = {0: 'No', 1: 'Yes', 2: 'Not sure'}
                main_formatted_values['Diabetes'] = diabetes_map.get(value, str(value))
            elif key == 'High_Cholesterol':
                hc_map = {0: 'No', 1: 'Yes', 2: 'Not sure'}
                main_formatted_values['High Cholesterol'] = hc_map.get(value, str(value))
            elif key == 'Family_History':
                fh_map = {0: 'No', 1: 'Yes', 2: 'Not sure'}
                main_formatted_values['Family History'] = fh_map.get(value, str(value))
            elif key == 'Chest_Pain':
                cp_freq_map = {0: 'Never', 1: 'Rarely', 2: 'Sometimes', 3: 'Often'}
                main_formatted_values['Chest Pain Frequency'] = cp_freq_map.get(value, str(value))
            elif key == 'Chest_Pain_Severity':
                cp_severity_map = {0: 'Mild', 1: 'Low-Moderate', 2: 'Moderate', 3: 'High-Moderate', 4: 'Severe'}
                main_formatted_values['Chest Pain Severity'] = cp_severity_map.get(value, str(value))
            elif key == 'Short_Breath':
                sb_freq_map = {0: 'Never', 1: 'Rarely', 2: 'Sometimes', 3: 'Often'}
                main_formatted_values['Shortness of Breath Frequency'] = sb_freq_map.get(value, str(value))
            elif key == 'Short_Breath_Duration':
                sb_duration_map = {0: '0 minute', 1: '1-5 minutes', 2: '6-15 minutes', 3: '16-30 minutes', 4: '31-60 minutes'}
                main_formatted_values['Shortness of Breath Duration'] = sb_duration_map.get(value, str(value))
            elif key == 'Exercise':
                exercise_map = {0: 'Never', 1: 'Rarely (1-2 times/month)', 2: 'Sometimes (1-2 times/week)', 3: 'Regularly (3+ times/week)'}
                main_formatted_values['Exercise Frequency'] = exercise_map.get(value, str(value))
            elif key == 'Fatty_Food':
                diet_map = {0: 'Rarely (less than weekly)', 1: 'Sometimes (1-2 times/week)', 2: 'Often (3-5 times/week)', 3: 'Very Often (daily)'}
                main_formatted_values['Diet Habits (Fried/Fatty Foods)'] = diet_map.get(value, str(value))
            elif key == 'Stress':
                stress_map = {0: 'Rarely (less than weekly)', 1: 'Sometimes (1-2 times/week)', 2: 'Often (3-5 times/week)', 3: 'Very Often (daily)'}
                main_formatted_values['Stress Level'] = stress_map.get(value, str(value))
            # Handle other numerical/string values that just need title casing and direct display
            elif key in ['patient_name', 'patient_contact', 'age', 'trestbps', 'chol', 'thalach', 'oldpeak', 
                        'height', 'weight', 'bmi', 'time_of_smoking', 'frequency_of_smoking', 'notes']:
                if isinstance(value, (int, float)) and not isinstance(value, bool): 
                    main_formatted_values[display_key] = float(value)
                else:
                    main_formatted_values[display_key] = value
            else:
                # For any other keys not specifically mapped, just use the cleaned-up key and original value
                main_formatted_values[display_key] = value


        # Get and process history entries
        history_entries = PredictionHistory.objects.filter(search_data=search_data).order_by('-created')
        processed_history = []
        for entry in history_entries:
            entry_parsed_data = {}
            if entry.values_list:
                try:
                    entry_parsed_data = json.loads(entry.values_list)
                except json.JSONDecodeError:
                    print(f"Error decoding history values_list JSON for PredictionHistory ID {entry.id}")
                    entry_parsed_data = {}

            entry_formatted_values = {}
            if isinstance(entry_parsed_data, dict):
                if 'features' in entry_parsed_data:
                    data_source_for_display_entry = {
                        'patient_name': entry_parsed_data.get('patient_name'),
                        'patient_contact': entry_parsed_data.get('patient_contact'),
                        **entry_parsed_data['features']
                    }
                else:
                    data_source_for_display_entry = entry_parsed_data
            else:
                print(f"Unexpected format for history values_list for PredictionHistory ID {entry.id}: {type(entry_parsed_data)}")
                data_source_for_display_entry = {}

            for key, value in data_source_for_display_entry.items():
                display_key = key.replace('_', ' ').title()

                if key == 'sex':
                    entry_formatted_values['Sex'] = 'Male' if value == 1.0 else 'Female'
                elif key == 'cp':
                    cp_map = {1.0: 'Typical Angina', 2.0: 'Atypical Angina', 3.0: 'Non-anginal Pain', 4.0: 'Asymptomatic'}
                    entry_formatted_values['Chest Pain Type'] = cp_map.get(value, str(value))
                elif key == 'fbs':
                    entry_formatted_values['Fasting Blood Sugar'] = 'Yes (>120 mg/dl)' if value == 1.0 else 'No (<=120 mg/dl)'
                elif key == 'restecg':
                    restecg_map = {0.0: 'Normal', 1.0: 'ST-T wave abnormality', 2.0: 'Left ventricular hypertrophy'}
                    entry_formatted_values['Resting ECG'] = restecg_map.get(value, str(value))
                elif key == 'exang':
                    entry_formatted_values['Exercise Induced Angina'] = 'Yes' if value == 1.0 else 'No'
                elif key == 'slope':
                    slope_map = {1.0: 'Upsloping', 2.0: 'Flat', 3.0: 'Downsloping'}
                    entry_formatted_values['Slope of Peak Exercise ST Segment'] = slope_map.get(value, str(value))
                elif key == 'ca':
                    entry_formatted_values['Number of Major Vessels'] = str(int(value)) if value is not None else 'N/A'
                elif key == 'thal':
                    thal_map = {1.0: 'Fixed Defect', 2.0: 'Normal', 3.0: 'Reversible Defect'}
                    entry_formatted_values['Thalassemia'] = thal_map.get(value, str(value))
                # New patient-specific mappings for history entries
                elif key == 'Gender':
                    entry_formatted_values['Gender'] = 'Male' if value == 1 else 'Female'
                elif key == 'Smoke':
                    entry_formatted_values['Smoking'] = 'Yes' if value == 1 else 'No'
                elif key == 'High_Blood_Pressure':
                    hp_map = {0: 'No', 1: 'Yes', 2: 'Not sure'}
                    entry_formatted_values['High Blood Pressure'] = hp_map.get(value, str(value))
                elif key == 'Diabetes':
                    diabetes_map = {0: 'No', 1: 'Yes', 2: 'Not sure'}
                    entry_formatted_values['Diabetes'] = diabetes_map.get(value, str(value))
                elif key == 'High_Cholesterol':
                    hc_map = {0: 'No', 1: 'Yes', 2: 'Not sure'}
                    entry_formatted_values['High Cholesterol'] = hc_map.get(value, str(value))
                elif key == 'Family_History':
                    fh_map = {0: 'No', 1: 'Yes', 2: 'Not sure'}
                    entry_formatted_values['Family History'] = fh_map.get(value, str(value))
                elif key == 'Chest_Pain':
                    cp_freq_map = {0: 'Never', 1: 'Rarely', 2: 'Sometimes', 3: 'Often'}
                    entry_formatted_values['Chest Pain Frequency'] = cp_freq_map.get(value, str(value))
                elif key == 'Chest_Pain_Severity':
                    cp_severity_map = {0: 'Mild', 1: 'Low-Moderate', 2: 'Moderate', 3: 'High-Moderate', 4: 'Severe'}
                    entry_formatted_values['Chest Pain Severity'] = cp_severity_map.get(value, str(value))
                elif key == 'Short_Breath':
                    sb_freq_map = {0: 'Never', 1: 'Rarely', 2: 'Sometimes', 3: 'Often'}
                    entry_formatted_values['Shortness of Breath Frequency'] = sb_freq_map.get(value, str(value))
                elif key == 'Short_Breath_Duration':
                    sb_duration_map = {0: '0 minute', 1: '1-5 minutes', 2: '6-15 minutes', 3: '16-30 minutes', 4: '31-60 minutes'}
                    entry_formatted_values['Shortness of Breath Duration'] = sb_duration_map.get(value, str(value))
                elif key == 'Exercise':
                    exercise_map = {0: 'Never', 1: 'Rarely (1-2 times/month)', 2: 'Sometimes (1-2 times/week)', 3: 'Regularly (3+ times/week)'}
                    entry_formatted_values['Exercise Frequency'] = exercise_map.get(value, str(value))
                elif key == 'Fatty_Food':
                    diet_map = {0: 'Rarely (less than weekly)', 1: 'Sometimes (1-2 times/week)', 2: 'Often (3-5 times/week)', 3: 'Very Often (daily)'}
                    entry_formatted_values['Diet Habits (Fried/Fatty Foods)'] = diet_map.get(value, str(value))
                elif key == 'Stress':
                    stress_map = {0: 'Rarely (less than weekly)', 1: 'Sometimes (1-2 times/week)', 2: 'Often (3-5 times/week)', 3: 'Very Often (daily)'}
                    entry_formatted_values['Stress Level'] = stress_map.get(value, str(value))
                # Handle other numerical/string values that just need title casing and direct display
                elif key in ['patient_name', 'patient_contact', 'age', 'trestbps', 'chol', 'thalach', 'oldpeak', 
                            'height', 'weight', 'bmi', 'time_of_smoking', 'frequency_of_smoking', 'notes']:
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        entry_formatted_values[display_key] = float(value)
                    else:
                        entry_formatted_values[display_key] = value
                else:
                    entry_formatted_values[display_key] = value

            entry.formatted_values = entry_formatted_values # Attach formatted values to the entry
            processed_history.append(entry)

        context = {
            'search_data': search_data,
            'history': processed_history,
            'main_formatted_values': main_formatted_values # Used for the 'Original Prediction' details
        }

        return render(request, 'prediction_history.html', context)

    except Search_Data.DoesNotExist:
        return render(request, 'prediction_history.html', {'error_message': 'Prediction record not found.'})
    except Exception as e:
        print(f"Error in view_prediction_history: {e}")
        return render(request, 'prediction_history.html', {'error_message': f'An error occurred: {e}'})

def train_patient_model():
    try:
        csv_path = 'media/medical_dataset.csv'  
        model_path = 'patient_model.pkl'
        metrics_path = 'patient_model_metrics.json' 
        
        # Load data
        df = pd.read_csv(csv_path)
        print("Patient data loaded, shape:", df.shape)
        
        # # Preprocessing: Map 'Gender' column
        # df['Gender'] = df['Gender'].map({'Male':0, 'Female':1})
        
        # Define features (X) and target (y) based on notebook
        X = df.drop('Result', axis=1)
        y = df['Result']
        
        # Get the list of feature names
        feature_names = X.columns.tolist()
        print("Features used:", feature_names)

        # Split data 
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print("Patient data split - train:", X_train.shape, "test:", X_test.shape)

        # Scale features (using StandardScaler) and train model (Logistic Regression)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        model = LogisticRegression(max_iter=1000)
        print("Training patient Logistic Regression model (scaled features, max_iter=1000)...")
        model.fit(X_train_scaled, y_train)
        
        # Evaluate model 
        accuracy = model.score(X_test_scaled, y_test)
        y_pred = model.predict(X_test_scaled)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        
        # Print detailed metrics to backend console
        print("=" * 60)
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
        
        # Save model, scalar, and metrics
        print("Saving patient model and metrics...")
        # joblib.dump(model, model_path)
        joblib.dump({'model': model, 'scaler': scaler}, model_path)
        
        metrics = {
            'accuracy': accuracy,
            'feature_names': feature_names,
            'confusion_matrix': cm.tolist(),
            'classification_report': report,
            'model_version': 'v1.0',
            'trained_on': str(datetime.now())
        }
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f)
        
        print("Patient model training completed successfully!")
        
    except FileNotFoundError:
        print(f"Error: Patient dataset not found at {csv_path}")
        raise 
    except Exception as e:
        print(f"Error in patient model training: {str(e)}")
        raise

def prdict_patient_heart_disease(patient_input_data: dict):
    try:
        model_path = 'patient_model.pkl'
        metrics_path = 'patient_model_metrics.json'
        
        # Check if model files exist
        if not os.path.exists(model_path) or not os.path.exists(metrics_path):
            print("Patient model files not found. Training model...")
            train_patient_model()
        
        # Load model and metrics
        try:
            loaded_data = joblib.load(model_path)
            model = loaded_data['model']
            scaler = loaded_data['scaler']
            # model = joblib.load(model_path) 2025/6/9 yo matra thyo
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
            feature_names = metrics.get('feature_names', [])
            accuracy = float(metrics.get('accuracy', 0.0)) * 100 # Convert accuracy to percentage
            print(f"Loaded patient model with accuracy: {accuracy:.2f}%")

            # Debugging: Print coefficients and scaler info for analysis
            if hasattr(model, 'coef_'):
                print("Model Coefficients (for positive class):")
                for i, name in enumerate(feature_names):
                    print(f"  {name}: {model.coef_[0][i]:.4f}")
            if hasattr(scaler, 'mean_') and hasattr(scaler, 'scale_'):
                print("Scaler Means and Scales:")
                for i, name in enumerate(feature_names):
                    print(f"  {name}: Mean={scaler.mean_[i]:.4f}, Scale={scaler.scale_[i]:.4f}")

        except Exception as e:
            print(f"Error loading patient model or metrics: {str(e)}. Retraining...")
            train_patient_model()
            # model = joblib.load(model_path) 2025/6/9 yo matra thyo
            loaded_data = joblib.load(model_path)
            model = loaded_data['model']
            scaler = loaded_data['scaler']
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
            feature_names = metrics.get('feature_names', [])
            accuracy = float(metrics.get('accuracy', 0.0)) * 100 # Convert accuracy to percentage
            print(f"Retrained and loaded patient model with accuracy: {accuracy:.2f}%")

        # Prepare input data for prediction by ensuring correct order and conversion to NumPy array
        input_data_list = [patient_input_data.get(feature, 0) for feature in feature_names]
        
        # Check for missing or invalid input (e.g., negative values where not expected)
        if any(val is None or (isinstance(val, (int, float)) and val < 0) for val in input_data_list):
            missing_or_invalid_features = [feature_names[i] for i, val in enumerate(input_data_list) if val is None or (isinstance(val, (int, float)) and val < 0)]
            raise ValueError(f"Missing or invalid input data for features: {missing_or_invalid_features}")

        # Convert to numpy array and reshape for single prediction
        X_predict = np.array(input_data_list).reshape(1, -1)
        
        # Scale features
        X_predict_scaled = scaler.transform(X_predict)
        
        # Make prediction and get probabilities
        pred = model.predict(X_predict_scaled)
        pred_proba = model.predict_proba(X_predict_scaled)[0]
        print("Patient prediction:", pred[0])
        print("Patient prediction probabilities:", pred_proba)
        
        # Print current model metrics for reference
        print("\n" + "=" * 50)
        print("CURRENT PATIENT MODEL METRICS")
        print("=" * 50)
        print(f"Model Accuracy: {accuracy:.2f}%")
        print("Model Type: Logistic Regression")
        print("Features Used: 19 patient parameters")
        
        # Load and display saved metrics if available
        try:
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
                if 'confusion_matrix' in metrics:
                    cm_saved = metrics['confusion_matrix']
                    print(f"Saved Confusion Matrix: TN={cm_saved[0][0]}, FP={cm_saved[0][1]}, FN={cm_saved[1][0]}, TP={cm_saved[1][1]}")
                    
                    # Calculate and display F1 scores from confusion matrix
                    tn, fp, fn, tp = cm_saved[0][0], cm_saved[0][1], cm_saved[1][0], cm_saved[1][1]
                    
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
                    
                    print("\nF1 SCORES (Calculated from Saved Confusion Matrix):")
                    print(f"F1 Score (Healthy Class 0): {f1_0:.4f}")
                    print(f"F1 Score (Unhealthy Class 1): {f1_1:.4f}")
                    print(f"F1 Score (Macro Average): {f1_macro:.4f}")
                    print(f"F1 Score (Weighted Average): {f1_weighted:.4f}")
                    
                if 'classification_report' in metrics:
                    print("\nCLASSIFICATION REPORT:")
                    print(metrics['classification_report'])
                if 'f1_scores' in metrics:
                    print("\nSAVED F1 SCORES:")
                    f1_scores = metrics['f1_scores']
                    for key, value in f1_scores.items():
                        print(f"F1 Score ({key}): {value:.4f}")
        except Exception as e:
            print(f"Could not load saved metrics: {e}")
        print("=" * 50)
        
        # Calculate feature importance/impact for explainability
        feature_impacts = []
        if hasattr(model, 'coef_') and model.coef_.shape[0] == 1:
            # For binary classification with a single output (Logistic Regression)
            coefficients = model.coef_[0]  # Get the coefficients for the positive class (1=unhealthy)
            
            # Get the mean and scale from the scaler for each feature
            feature_means = scaler.mean_
            feature_scales = scaler.scale_
            
            # Calculate total log-odds contribution using medical knowledge
            total_log_odds = 0
            for i, feature_name in enumerate(feature_names):
                original_val = patient_input_data.get(feature_name, 0)
                
                # Calculate impact using the same medical knowledge logic
                if feature_name == 'Age':
                    if original_val >= 65:
                        log_odds_contribution = 0.5
                    elif original_val >= 50:
                        log_odds_contribution = 0.3
                    elif original_val >= 35:
                        log_odds_contribution = 0.1
                    else:
                        log_odds_contribution = -0.1
                elif feature_name == 'Gender':
                    log_odds_contribution = 0.2 if original_val == 0 else -0.1
                elif feature_name == 'BMI':
                    if original_val >= 30:
                        log_odds_contribution = 0.4
                    elif original_val >= 25:
                        log_odds_contribution = 0.2
                    else:
                        log_odds_contribution = -0.1
                elif feature_name == 'Smoke':
                    log_odds_contribution = 0.4 if original_val == 1 else -0.1
                elif feature_name == 'Time_of_Smoking':
                    if original_val >= 20:
                        log_odds_contribution = 0.5
                    elif original_val >= 10:
                        log_odds_contribution = 0.3
                    elif original_val >= 5:
                        log_odds_contribution = 0.2
                    else:
                        log_odds_contribution = 0.0
                elif feature_name == 'High_Blood_Pressure':
                    log_odds_contribution = 0.4 if original_val == 1 else -0.1
                elif feature_name == 'Diabetes':
                    log_odds_contribution = 0.5 if original_val == 1 else -0.1
                elif feature_name == 'High_Cholesterol':
                    log_odds_contribution = 0.3 if original_val == 1 else -0.1
                elif feature_name == 'Family_History':
                    log_odds_contribution = 0.3 if original_val == 1 else -0.1
                elif feature_name == 'Chest_Pain':
                    if original_val >= 3:
                        log_odds_contribution = 0.5
                    elif original_val >= 2:
                        log_odds_contribution = 0.3
                    elif original_val >= 1:
                        log_odds_contribution = 0.2
                    else:
                        log_odds_contribution = 0.0
                elif feature_name == 'Chest_Pain_Severity':
                    if original_val >= 4:
                        log_odds_contribution = 0.5
                    elif original_val >= 3:
                        log_odds_contribution = 0.4
                    elif original_val >= 2:
                        log_odds_contribution = 0.3
                    else:
                        log_odds_contribution = 0.0
                elif feature_name == 'Short_Breath':
                    if original_val >= 3:
                        log_odds_contribution = 0.4
                    elif original_val >= 2:
                        log_odds_contribution = 0.3
                    elif original_val >= 1:
                        log_odds_contribution = 0.2
                    else:
                        log_odds_contribution = 0.0
                elif feature_name == 'Exercise':
                    log_odds_contribution = 0.3 if original_val == 0 else -0.2
                elif feature_name == 'Fatty_Food':
                    if original_val >= 4:
                        log_odds_contribution = 0.4
                    elif original_val >= 3:
                        log_odds_contribution = 0.3
                    elif original_val >= 2:
                        log_odds_contribution = 0.2
                    else:
                        log_odds_contribution = 0.0
                elif feature_name == 'Stress':
                    if original_val >= 4:
                        log_odds_contribution = 0.4
                    elif original_val >= 3:
                        log_odds_contribution = 0.3
                    elif original_val >= 2:
                        log_odds_contribution = 0.2
                    else:
                        log_odds_contribution = 0.0
                else:
                    # For other features, use reduced model coefficient
                    log_odds_contribution = ((original_val - feature_means[i]) / feature_scales[i]) * coefficients[i] * 0.5
                
                total_log_odds += log_odds_contribution
            
            for i, feature_name in enumerate(feature_names):
                # Original value from patient_input_data for context
                original_val = patient_input_data.get(feature_name, 0)
                
                # Calculate impact based on medical knowledge rather than flawed model coefficients
                # This ensures medically accurate risk assessment
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
                    if original_val == 0:  # Male
                        impact_score = 0.2  # Moderate risk
                    else:  # Female
                        impact_score = -0.1  # Slightly protective
                        
                elif feature_name == 'BMI':
                    if original_val >= 30:
                        impact_score = 0.4  # Strong risk (obese)
                    elif original_val >= 25:
                        impact_score = 0.2  # Moderate risk (overweight)
                    else:
                        impact_score = -0.1  # Protective
                        
                elif feature_name == 'Smoke':
                    if original_val == 1:
                        impact_score = 0.4  # Strong risk
                    else:
                        impact_score = -0.1  # Protective
                        
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
                    if original_val == 1:
                        impact_score = 0.4  # Strong risk
                    else:
                        impact_score = -0.1  # Protective
                        
                elif feature_name == 'Diabetes':
                    if original_val == 1:
                        impact_score = 0.5  # Strong risk
                    else:
                        impact_score = -0.1  # Protective
                        
                elif feature_name == 'High_Cholesterol':
                    if original_val == 1:
                        impact_score = 0.3  # Moderate risk
                    else:
                        impact_score = -0.1  # Protective
                        
                elif feature_name == 'Family_History':
                    if original_val == 1:
                        impact_score = 0.3  # Moderate risk
                    else:
                        impact_score = -0.1  # Protective
                        
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
                    if original_val == 0:  # No exercise
                        impact_score = 0.3  # Moderate risk
                    else:
                        impact_score = -0.2  # Protective
                        
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
                    # For other features, use the model coefficient but with reduced weight
                    impact_score = ((original_val - feature_means[i]) / feature_scales[i]) * coefficients[i] * 0.5
                
                # Calculate relative importance as percentage of total log-odds contribution
                if total_log_odds != 0:
                    relative_importance = abs(impact_score) / abs(total_log_odds) * 100
                else:
                    relative_importance = 0.0
                
                # Normalize impact to [-1, 1] range using sigmoid scaling
                # This makes the impact more interpretable
                normalized_impact = 2 * (1 / (1 + np.exp(-impact_score))) - 1
                
                feature_impacts.append({
                    'feature': feature_name,
                    'value': original_val,
                    'coefficient': round(coefficients[i], 3),
                    'impact': round(impact_score, 3),
                    'normalized_impact': round(normalized_impact, 3),
                    'relative_importance': round(relative_importance, 1)  # Convert to percentage
                })
        
        # Use standard threshold for balanced classification
        # If the probability of being unhealthy (pred_proba[1]) is > 0.5, predict 1 (unhealthy)
        # Otherwise, predict 0 (healthy)
        pred_value = 1 if pred_proba[1] > 0.5 else 0 # Standard threshold
        
        return accuracy, pred_value, pred_proba, feature_impacts
        
    except Exception as e:
        print(f"Error in patient prediction: {str(e)}")
        raise

    #     # Prepare input data for prediction
    #     input_data_list = [patient_input_data.get(feature) for feature in feature_names]
        
    #     # Check for missing input features
    #     if None in input_data_list:
    #         missing_features = [feature_names[i] for i, val in enumerate(input_data_list) if val is None]
    #         raise ValueError(f"Missing input data for features: {missing_features}")

    #     # Convert to numpy array and reshape for prediction
    #     X_predict = np.array(input_data_list).reshape(1, -1)
        
    #     # Make prediction and get probabilities
    #     pred = model.predict(X_predict)
    #     pred_proba = model.predict_proba(X_predict)[0]
    #     print("Patient prediction:", pred[0])
    #     print("Patient prediction probabilities:", pred_proba)
        
    #     return accuracy, pred[0], pred_proba # Return accuracy, prediction, and probabilities
        
    # except Exception as e:
    #     print(f"Error in patient prediction: {str(e)}")
    #     raise 2025/6/9 yo thyo

def extract_city(address):
    """
    Extract city from address string.
    Assumes city is the last part of the address after any commas.
    """
    if not address:
        return ""
    # Split by comma and get the last part, then strip whitespace
    parts = [part.strip() for part in address.split(',')]
    return parts[-1].strip()

@login_required(login_url="login")
def add_heartdetail_patient(request):
    patient_name = ""
    patient_age = ""
    try:
        patient = Patient.objects.get(user=request.user)
        patient_name = f"{patient.user.first_name} {patient.user.last_name}".strip()
        if patient.dob:
            today = date.today()
            patient_age = today.year - patient.dob.year - ((today.month, today.day) < (patient.dob.month, patient.dob.day))
    except Patient.DoesNotExist:
        pass

    if request.method == "POST":
        # Collect data from the form and map to the features expected by the patient model
        age = int(request.POST.get('age', 0))
        sex_form = int(request.POST.get('sex',0)) # 0 for Female, 1 for Male in form
        gender_for_model = 0 if sex_form == 1 else 1 # Model expects 0=Male, 1=Female (opposite of form)

        height = float(request.POST.get('height', 0))
        weight = float(request.POST.get('weight', 0))
        bmi = round(weight / ((height / 100) ** 2), 2) if height > 0 else 0

        smoke = int(request.POST.get('smoking', 0)) 
        time_of_smoking = int(request.POST.get('time_of_smoking', 0))
        frequency_of_smoking = int(request.POST.get('frequency_of_smoking', 0))
        
        high_blood_pressure = 1 if request.POST.get('hypertension', '0') == '1' else 0
        diabetes = 1 if int(request.POST.get('diabetes', 2)) == 1 else 0
        high_cholesterol = 1 if request.POST.get('high_cholesterol', '0') == '1' else 0
        family_history = int(request.POST.get('family_history', 0))

        chest_pain = int(request.POST.get('chest_pain', 0))
        chest_pain_severity = int(request.POST.get('chest_pain_severity', 0))

        short_breath = int(request.POST.get('shortness_of_breath', 0))
        short_breath_duration = int(request.POST.get('shortness_of_breath_duration', 0))

        exercise = int(request.POST.get('physical_activity', 0))
        fatty_food = int(request.POST.get('diet_habits', 0))
        stress = int(request.POST.get('stress_level', 0)) 

        # Create a dictionary with the exact feature names expected by the model
        patient_input_data_for_prediction = {
            'Age': age,
            'Gender': gender_for_model,
            'Height': height,
            'Weight': weight,
            'BMI': bmi,
            'Smoke': smoke,
            'Time_of_Smoking': time_of_smoking,
            'Frequency_of_smoking': frequency_of_smoking,
            'High_Blood_Pressure': high_blood_pressure,
            'Diabetes': diabetes,
            'High_Cholesterol': high_cholesterol,
            'Family_History': family_history,
            'Chest_Pain': chest_pain,
            'Chest_Pain_Severity': chest_pain_severity,
            'Short_Breath': short_breath,
            'Short_Breath_Duration': short_breath_duration,
            'Exercise': exercise,
            'Fatty_Food': fatty_food,
            'Stress': stress,
        }
        
        print("Patient model input dictionary (matching trained model features):", patient_input_data_for_prediction)
        
        # Call the patient prediction function with the prepared dictionary
        accuracy, pred_value, pred_proba, feature_impacts = prdict_patient_heart_disease(patient_input_data_for_prediction)

        # Save prediction data
        search_data = None
        patient = None
        doctor = None
        try:
            patient = Patient.objects.get(user=request.user)
        except Patient.DoesNotExist:
            try:
                doctor = Doctor.objects.get(user=request.user)
            except Doctor.DoesNotExist:
                pass

        if patient or doctor:
            search_data = Search_Data.objects.create(
                patient=patient,
                doctor=doctor,
                prediction_accuracy=accuracy,
                result=int(pred_value), # Ensure result is int
                values_list=json.dumps(patient_input_data_for_prediction), # Save the input dictionary as JSON string
                feature_impacts=json.dumps(feature_impacts) # Save feature impacts
            )
            # Create history entry
            if search_data:
                PredictionHistory.objects.create(
                    search_data=search_data,
                    prediction_accuracy=accuracy,
                    result=int(pred_value), # Ensure result is int
                    values_list=json.dumps(patient_input_data_for_prediction), # Save the input dictionary as JSON string
                    feature_impacts=json.dumps(feature_impacts) # Save feature impacts
                )

        print(f"Patient Final prediction: {pred_value} (0=healthy, 1=unhealthy)")
        print(f"Patient Confidence: healthy={pred_proba[0]:.1f}%, unhealthy={pred_proba[1]:.1f}%")
        
        # Return JSON if AJAX request
        if request.headers.get('x-requested-with') == 'XMLHttpRequest':
            return JsonResponse({
                'prediction': int(pred_value),
                'accuracy': round(float(accuracy), 2),
                'healthy_prob': float(pred_proba[0]),
                'unhealthy_prob': float(pred_proba[1]),
                'feature_impacts': feature_impacts
            })
        # Instead of redirecting, render the template directly
        pred_text = "<span class='healthy'>Healthy</span>" if int(pred_value) == 0 else "<span class='unhealthy'>Unhealthy</span>"
        
        # Format accuracy to two decimal places
        formatted_accuracy = round(float(accuracy), 2)

        # Pre-process feature_impacts for display in the template
        if feature_impacts:
            for item in feature_impacts:
                item['feature'] = item['feature'].replace('_', ' ')

        # Get patient's city
        patient_city = extract_city(patient.address) if patient else ""
        
        # Find doctors in the same city who are authorized (status=1)
        doctors = Doctor.objects.filter(
            status=1,  # Only authorized doctors
            address__icontains=patient_city  # Match by city
        ).order_by('category')  # Order by specialization
        
        # If no doctors found in exact city, try nearby cities (you can expand this logic)
        if not doctors.exists() and patient_city:
            # Try to find doctors in nearby cities (this is a simple example)
            # You could expand this with a proper city database and distance calculation
            doctors = Doctor.objects.filter(
                status=1,
                address__icontains=patient_city.split()[0]  # Try matching first word of city
            ).order_by('category')
        
        return render(request, 'predict_disease.html', {
            'pred': pred_text, 
            'accuracy': formatted_accuracy, 
            'doctor': doctors,
            'patient_city': patient_city,
            'feature_impacts': feature_impacts
        })
        
    return render(request, 'add_heartdetail_patient.html', {'patient_name': patient_name, 'patient_age': patient_age})
