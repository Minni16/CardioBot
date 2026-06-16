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
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from django.http import HttpResponse, JsonResponse
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report, f1_score,
                             roc_curve, auc, precision_recall_curve, average_precision_score, log_loss)
from sklearn.model_selection import learning_curve
from sklearn.linear_model import SGDClassifier

from datetime import datetime, date

# Bumped when training recipe or saved metrics schema changes (triggers retrain on load).
DOCTOR_HEART_MODEL_RECIPE_VERSION = 3


def normalize_doctor_heart_cp(value):
    """Map chest-pain type to media/heart.csv encoding (0–3). Legacy UCI form used 1–4."""
    v = int(value)
    if 1 <= v <= 4:
        return v - 1
    if 0 <= v <= 3:
        return v
    raise ValueError("cp must be 0–3 (dataset) or legacy 1–4 (UCI labels)")


def normalize_doctor_heart_slope(value):
    """Map ST slope to media/heart.csv encoding (0–2). Legacy form used 1–3 (UCI)."""
    v = int(value)
    if 1 <= v <= 3:
        return v - 1
    if 0 <= v <= 2:
        return v
    raise ValueError("slope must be 0–2 (dataset) or legacy 1–3 (UCI labels)")


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
                license_doc = request.FILES.get('license_document')

                # Validate contact number
                if not con.isdigit() or len(con) != 10:
                    error = "invalid_contact"
                    return render(request, 'login.html', {'error': error, 'show_signup': True})

                # Doctors must submit a license/certificate document for admin review
                if type == "Doctor" and not license_doc:
                    error = "license_required"
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
                        Doctor.objects.create(dob=d, image=im, user=user, contact=con, address=add, status=2, license_document=license_doc)
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
        o = request.POST['pwd3']
        n = request.POST['pwd1']
        c = request.POST['pwd2']
        if not request.user.check_password(o):
            terror = "wrong_old"
        elif c == n:
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


def generate_model_plots(prefix, X_train, X_test, y_train, y_test,
                          lr_model, rf_model, scaler, feature_names, best_C=1.0):
    """
    Generates and saves all evaluation plots for both LR and RF models.
    prefix: 'doctor' or 'patient'
    Saves PNGs to media/model_plots/ and prints a summary to terminal.
    Returns dict of {plot_name: file_path}.
    """
    plt.switch_backend('Agg')
    plots_dir = os.path.join('media', 'model_plots')
    os.makedirs(plots_dir, exist_ok=True)

    X_tr_s = scaler.transform(X_train)
    X_te_s = scaler.transform(X_test)

    lr_pred  = lr_model.predict(X_te_s)
    lr_proba = lr_model.predict_proba(X_te_s)[:, 1]
    rf_pred  = rf_model.predict(X_test)
    rf_proba = rf_model.predict_proba(X_test)[:, 1]

    saved = {}
    title_prefix = prefix.title()

    def _save(fig, name):
        p = os.path.join(plots_dir, f'{prefix}_{name}.png')
        fig.savefig(p, dpi=120, bbox_inches='tight')
        plt.close(fig)
        saved[name] = p
        print(f"  [plot] {p}")

    # ── 1. Confusion Matrices ──────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f'{title_prefix} Model — Confusion Matrices', fontsize=13, fontweight='bold')
    for ax, pred, title in [(axes[0], lr_pred, 'Logistic Regression'),
                             (axes[1], rf_pred, 'Random Forest')]:
        cm_vals = confusion_matrix(y_test, pred)
        sns.heatmap(cm_vals, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Healthy', 'Disease'],
                    yticklabels=['Healthy', 'Disease'], linewidths=0.5)
        ax.set(title=title, xlabel='Predicted', ylabel='Actual')
    plt.tight_layout()
    _save(fig, 'confusion_matrix')

    # ── 2. ROC Curve (Prediction Curve) ───────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 6))
    for proba, label, color in [(lr_proba, 'Logistic Regression', '#e74c3c'),
                                  (rf_proba, 'Random Forest', '#2e86de')]:
        fpr, tpr, _ = roc_curve(y_test, proba)
        ax.plot(fpr, tpr, color=color, lw=2, label=f'{label} (AUC={auc(fpr, tpr):.3f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random classifier')
    ax.set(xlabel='False Positive Rate', ylabel='True Positive Rate',
           title=f'{title_prefix} Model — ROC / Prediction Curve')
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)
    _save(fig, 'roc_curve')

    # ── 3. Precision-Recall Curve ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 6))
    for proba, label, color in [(lr_proba, 'Logistic Regression', '#e74c3c'),
                                  (rf_proba, 'Random Forest', '#2e86de')]:
        prec, rec, _ = precision_recall_curve(y_test, proba)
        ap = average_precision_score(y_test, proba)
        ax.plot(rec, prec, color=color, lw=2, label=f'{label} (AP={ap:.3f})')
    ax.set(xlabel='Recall', ylabel='Precision',
           title=f'{title_prefix} Model — Precision-Recall Curve')
    ax.legend(loc='upper right')
    ax.grid(alpha=0.3)
    _save(fig, 'pr_curve')

    # ── 4. Learning / Validation Curves ───────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'{title_prefix} Model — Learning / Validation Curves', fontsize=13, fontweight='bold')
    lr_pipe_lc = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(C=best_C, class_weight='balanced',
                                   max_iter=4000, random_state=42, solver='lbfgs'))
    ])
    rf_lc = RandomForestClassifier(n_estimators=100, max_depth=12, min_samples_leaf=2,
                                    class_weight='balanced', random_state=42, n_jobs=1)
    for ax, estimator, X_data, label in [
        (axes[0], lr_pipe_lc, X_train, 'Logistic Regression'),
        (axes[1], rf_lc,      X_train, 'Random Forest'),
    ]:
        try:
            tr_sz, tr_sc, cv_sc = learning_curve(
                estimator, X_data, y_train, cv=5, scoring='accuracy',
                train_sizes=np.linspace(0.1, 1.0, 10), random_state=42, n_jobs=1
            )
            tr_m, tr_s = tr_sc.mean(1), tr_sc.std(1)
            cv_m, cv_s = cv_sc.mean(1), cv_sc.std(1)
            ax.plot(tr_sz, tr_m, 'o-', color='#e74c3c', label='Training accuracy')
            ax.fill_between(tr_sz, tr_m - tr_s, tr_m + tr_s, alpha=0.15, color='#e74c3c')
            ax.plot(tr_sz, cv_m, 'o-', color='#2e86de', label='CV accuracy')
            ax.fill_between(tr_sz, cv_m - cv_s, cv_m + cv_s, alpha=0.15, color='#2e86de')
            ax.set_ylim(0.4, 1.05)
        except Exception as e:
            ax.text(0.5, 0.5, str(e), ha='center', va='center', transform=ax.transAxes, fontsize=8)
        ax.set(xlabel='Training samples', ylabel='Accuracy', title=label)
        ax.legend()
        ax.grid(alpha=0.3)
    plt.tight_layout()
    _save(fig, 'learning_curve')

    # ── 5. LR Loss Curve (epoch-by-epoch via SGD) ─────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    try:
        sgd_clf = SGDClassifier(loss='log_loss', random_state=42,
                                 class_weight='balanced', tol=None, max_iter=1)
        tr_losses, vl_losses = [], []
        for _ in range(100):
            sgd_clf.partial_fit(X_tr_s, y_train, classes=np.unique(y_train))
            tr_losses.append(log_loss(y_train, sgd_clf.predict_proba(X_tr_s)))
            vl_losses.append(log_loss(y_test,  sgd_clf.predict_proba(X_te_s)))
        ax.plot(range(1, 101), tr_losses, color='#e74c3c', lw=2, label='Training loss')
        ax.plot(range(1, 101), vl_losses, color='#2e86de', lw=2, label='Validation loss')
        ax.set(xlabel='Epoch', ylabel='Log Loss',
               title=f'{title_prefix} Model — Logistic Regression Loss Curve (SGD)')
        ax.legend()
        ax.grid(alpha=0.3)
    except Exception as e:
        ax.text(0.5, 0.5, f'Loss curve error:\n{e}', ha='center', va='center',
                transform=ax.transAxes, fontsize=9)
    _save(fig, 'lr_loss_curve')

    # ── 6. RF OOB Error Curve (equivalent of loss/epoch for RF) ───────────
    fig, ax = plt.subplots(figsize=(8, 5))
    try:
        n_range = list(range(10, 310, 10))
        rf_oob = RandomForestClassifier(max_depth=12, min_samples_leaf=2,
                                         class_weight='balanced', random_state=42,
                                         n_jobs=1, oob_score=True, warm_start=True,
                                         n_estimators=10)
        oob_errors = []
        for n in n_range:
            rf_oob.set_params(n_estimators=n)
            rf_oob.fit(X_train, y_train)
            oob_errors.append(1 - rf_oob.oob_score_)
        ax.plot(n_range, oob_errors, color='#2e86de', lw=2, marker='o', markersize=4)
        min_idx = int(np.argmin(oob_errors))
        ax.axvline(n_range[min_idx], color='#e74c3c', ls='--', lw=1.5,
                   label=f'Best: {n_range[min_idx]} trees (OOB={oob_errors[min_idx]:.4f})')
        ax.set(xlabel='Number of Trees (n_estimators)', ylabel='OOB Error Rate',
               title=f'{title_prefix} Model — Random Forest OOB Error Curve')
        ax.legend()
        ax.grid(alpha=0.3)
    except Exception as e:
        ax.text(0.5, 0.5, f'OOB curve error:\n{e}', ha='center', va='center',
                transform=ax.transAxes, fontsize=9)
    _save(fig, 'rf_oob_curve')

    # ── 7. Feature Analysis (LR coefficients + RF importances) ────────────
    n_feat = len(feature_names)
    fig, axes = plt.subplots(1, 2, figsize=(16, max(5, n_feat * 0.45)))
    fig.suptitle(f'{title_prefix} Model — Feature Analysis', fontsize=13, fontweight='bold')
    # LR coefficients
    coef = lr_model.coef_[0]
    idx_lr = np.argsort(np.abs(coef))
    colors_lr = ['#e74c3c' if c > 0 else '#27ae60' for c in coef[idx_lr]]
    axes[0].barh([feature_names[i] for i in idx_lr], coef[idx_lr], color=colors_lr)
    axes[0].axvline(0, color='black', lw=0.8)
    axes[0].set(title='LR Coefficients  (red = raises risk, green = lowers risk)',
                xlabel='Coefficient value')
    axes[0].grid(axis='x', alpha=0.3)
    # RF feature importances
    importances = rf_model.feature_importances_
    idx_rf = np.argsort(importances)
    axes[1].barh([feature_names[i] for i in idx_rf], importances[idx_rf], color='#2e86de')
    axes[1].set(title='RF Feature Importances', xlabel='Importance')
    axes[1].grid(axis='x', alpha=0.3)
    plt.tight_layout()
    _save(fig, 'feature_analysis')

    # ── Terminal summary ───────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"ALL PLOTS SAVED FOR {title_prefix.upper()} MODEL")
    print(f"{'='*60}")
    for k, v in saved.items():
        print(f"  {k:25s} -> {v}")

    return saved


def retrain_heart_model():
    """
    Train doctor (13-feature) heart model on media/heart.csv.
    Deduplicates rows, tunes LogisticRegression C via stratified CV, fits on all clean rows,
    and stores CV mean accuracy as the primary reported metric.
    """
    try:
        csv_path = 'media/heart.csv'
        model_path = 'heart_model.pkl'
        acc_path = 'heart_model_acc.txt'
        scaler_path = 'heart_model_scaler.pkl'

        df = pd.read_csv(csv_path)
        print("Data loaded, shape:", df.shape)

        required_columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                            'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']

        if not all(col in df.columns for col in required_columns):
            raise ValueError("Missing required columns in the dataset")

        df = df.dropna(subset=required_columns)
        n_before = len(df)
        df = df.drop_duplicates()
        print(f"After dropna + dedupe: {len(df)} rows (removed {n_before - len(df)} duplicate / incomplete rows)")

        print("Class distribution:")
        print(df['target'].value_counts(normalize=True) * 100)

        X = df[['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']]
        y = df['target']

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        param_grid = {'clf__C': [0.1, 0.25, 0.5, 1.0, 2.0, 4.0]}
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(
                max_iter=4000,
                class_weight='balanced',
                random_state=42,
                solver='lbfgs',
            )),
        ])
        print("Tuning LogisticRegression (5-fold stratified CV)...")
        grid = GridSearchCV(
            pipe, param_grid, cv=cv, scoring='accuracy', n_jobs=1, refit=True
        )
        grid.fit(X, y)
        best_cv = float(grid.best_score_)
        best_C = float(grid.best_params_['clf__C'])
        print(f"Best CV mean accuracy: {best_cv * 100:.2f}% (C={best_C})")

        final_pipe = grid.best_estimator_
        model = final_pipe.named_steps['clf']
        scaler = final_pipe.named_steps['scaler']

        # Holdout confusion matrix / report (same split seed as before for stability)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        scaler_h = StandardScaler().fit(X_train)
        X_tr_s = scaler_h.transform(X_train)
        X_te_s = scaler_h.transform(X_test)
        model_h = LogisticRegression(
            max_iter=4000,
            class_weight='balanced',
            random_state=42,
            solver='lbfgs',
            C=best_C,
        )
        model_h.fit(X_tr_s, y_train)
        y_pred = model_h.predict(X_te_s)
        holdout_acc = float(accuracy_score(y_test, y_pred))
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred)

        # --- Random Forest: same train/test split for apples-to-apples terminal comparison ---
        rf = RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42,
            n_jobs=1,
        )
        rf.fit(X_train, y_train)
        y_pred_rf = rf.predict(X_test)
        holdout_acc_rf = float(accuracy_score(y_test, y_pred_rf))
        cm_rf = confusion_matrix(y_test, y_pred_rf)
        report_rf = classification_report(y_test, y_pred_rf)

        print("=" * 60)
        print("DOCTOR MODEL COMPARISON (same 80/20 stratified holdout)")
        print("=" * 60)
        print("\n--- Logistic Regression (scaled features, tuned C) ---")
        print(f"Holdout accuracy: {holdout_acc * 100:.2f}%")
        print(f"CV mean accuracy (saved primary metric): {best_cv * 100:.2f}%")
        print("\nCONFUSION MATRIX (Logistic Regression):")
        print(f"TN={cm[0][0]}, FP={cm[0][1]}, FN={cm[1][0]}, TP={cm[1][1]}")
        print("\nCLASSIFICATION REPORT (Logistic Regression):")
        print(report)

        print("\n" + "-" * 60)
        print("--- Random Forest (raw features, same split) ---")
        print(f"Holdout accuracy: {holdout_acc_rf * 100:.2f}%")
        print("\nCONFUSION MATRIX (Random Forest):")
        print(f"TN={cm_rf[0][0]}, FP={cm_rf[0][1]}, FN={cm_rf[1][0]}, TP={cm_rf[1][1]}")
        print("\nCLASSIFICATION REPORT (Random Forest):")
        print(report_rf)

        print("\n" + "-" * 60)
        print("SUMMARY (holdout only; production API still uses Logistic Regression)")
        print(
            f"  Logistic Regression holdout accuracy: {holdout_acc * 100:.2f}%  |  "
            f"Random Forest holdout accuracy: {holdout_acc_rf * 100:.2f}%"
        )
        if holdout_acc_rf > holdout_acc:
            print(f"  On this split, Random Forest is +{(holdout_acc_rf - holdout_acc) * 100:.2f} pp ahead.")
        elif holdout_acc_rf < holdout_acc:
            print(f"  On this split, Logistic Regression is +{(holdout_acc - holdout_acc_rf) * 100:.2f} pp ahead.")
        else:
            print("  Both models tie on holdout accuracy for this split.")
        print("=" * 60)

        f1_class_0 = f1_score(y_test, y_pred, pos_label=0)
        f1_class_1 = f1_score(y_test, y_pred, pos_label=1)
        f1_macro = f1_score(y_test, y_pred, average='macro')
        f1_weighted = f1_score(y_test, y_pred, average='weighted')

        # ── Winner selection ───────────────────────────────────────────────
        if holdout_acc_rf >= holdout_acc:
            winner_model = rf
            winner_type  = 'RandomForest'
            winner_acc   = holdout_acc_rf
            loser_type   = 'LogisticRegression'
        else:
            winner_model = model_h
            winner_type  = 'LogisticRegression'
            winner_acc   = holdout_acc
            loser_type   = 'RandomForest'

        print(f"\n*** WINNER: {winner_type} (holdout={winner_acc*100:.2f}%) saved as production model ***")
        print(f"    ({loser_type} retained in terminal for analysis only)")

        # ── Save production bundle + backward-compat scaler ────────────────
        print("Saving model and scaler...")
        prod_bundle = {'model': winner_model, 'model_type': winner_type, 'scaler': scaler_h}
        joblib.dump(prod_bundle, model_path)
        joblib.dump(scaler_h, scaler_path)

        # ── Generate all evaluation plots ──────────────────────────────────
        feature_cols_list = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                              'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
        try:
            generate_model_plots('doctor', X_train, X_test, y_train, y_test,
                                  model_h, rf, scaler_h, feature_cols_list, best_C=best_C)
        except Exception as plot_err:
            print(f"[WARN] Plot generation failed: {plot_err}")

        winner_cm_doc  = cm_rf   if winner_type == 'RandomForest' else cm
        winner_rep_doc = report_rf if winner_type == 'RandomForest' else report
        metrics = {
            'accuracy': best_cv,
            'holdout_accuracy': holdout_acc,
            'holdout_accuracy_rf': holdout_acc_rf,
            'best_model_type': winner_type,
            'best_C': best_C,
            'cv_folds': 5,
            'confusion_matrix': winner_cm_doc.tolist(),
            'confusion_matrix_lr': cm.tolist(),
            'confusion_matrix_rf': cm_rf.tolist(),
            'classification_report': winner_rep_doc,
            'classification_report_lr': report,
            'classification_report_rf': report_rf,
            'feature_names': X.columns.tolist(),
            'trained_on': str(datetime.now()),
            'model_type': winner_type,
            'recipe_version': DOCTOR_HEART_MODEL_RECIPE_VERSION,
            'training_rows': len(df),
            'f1_scores': {
                'healthy': float(f1_class_0),
                'disease': float(f1_class_1),
                'macro': float(f1_macro),
                'weighted': float(f1_weighted),
            },
        }
        with open(acc_path, 'w') as f:
            json.dump(metrics, f, indent=4)

        print("Model training completed successfully!")
        return winner_model, scaler_h, best_cv

    except Exception as e:
        print(f"Error in model training: {str(e)}")
        raise

def prdict_heart_disease(list_data):
    try:
        model_path = 'heart_model.pkl'
        scaler_path = 'heart_model_scaler.pkl'
        acc_path = 'heart_model_acc.txt'

        accuracy = None
        model = None
        scaler = None

        def _metrics_need_retrain(raw_content):
            try:
                m = json.loads(raw_content)
            except json.JSONDecodeError:
                return True
            if not isinstance(m, dict):
                return True
            return int(m.get('recipe_version', 0)) != DOCTOR_HEART_MODEL_RECIPE_VERSION

        def _load_doctor_bundle():
            raw = joblib.load(model_path)
            if isinstance(raw, dict):
                return raw['model'], raw.get('model_type', 'LogisticRegression'), raw['scaler']
            # legacy: bare model object + separate scaler file
            return raw, 'LogisticRegression', joblib.load(scaler_path)

        model_type = 'LogisticRegression'
        if not all(os.path.exists(p) for p in (model_path, scaler_path, acc_path)):
            print("Model files not found. Retraining model...")
            retrain_heart_model()
            model, model_type, scaler = _load_doctor_bundle()
            with open(acc_path, 'r') as f:
                metrics = json.load(f)
            accuracy = float(metrics.get('accuracy', 0.0)) * 100
        else:
            try:
                with open(acc_path, 'r') as f:
                    content = f.read()
                stale = _metrics_need_retrain(content)
                model, model_type, scaler = _load_doctor_bundle()
                if stale:
                    raise ValueError('recipe_outdated')
                metrics = json.loads(content)
                accuracy = float(metrics.get('accuracy', 0.0)) * 100
            except Exception as e:
                print(f"Error loading model or stale metrics ({e}). Retraining...")
                retrain_heart_model()
                model, model_type, scaler = _load_doctor_bundle()
                with open(acc_path, 'r') as f:
                    metrics = json.load(f)
                accuracy = float(metrics.get('accuracy', 0.0)) * 100
        
        # Ensure input is in correct format
        if not isinstance(list_data, (list, np.ndarray)):
            raise ValueError("Input must be a list or numpy array")
            
        if len(list_data) != 13:
            raise ValueError(f"Expected 13 features, got {len(list_data)}")

        feature_cols = [
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
            'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal',
        ]
        X = pd.DataFrame([list_data], columns=feature_cols)
        
        # Print input data for debugging
        print("Input data:", list_data)
        
        # Scale features (RF uses raw; LR uses scaled)
        if model_type == 'RandomForest':
            X_input = X
        else:
            X_input = pd.DataFrame(scaler.transform(X), columns=X.columns)
        print("Model type in use:", model_type)

        # Make prediction
        pred = model.predict(X_input)
        pred_proba = model.predict_proba(X_input)[0]
        print("Prediction:", pred[0])
        print("Prediction probabilities:", pred_proba)
        
        # Print current model metrics for reference
        print("\n" + "=" * 50)
        print("CURRENT HEART DISEASE MODEL METRICS")
        print("=" * 50)
        print(f"Model Accuracy: {accuracy:.2f}%")
        print(f"Model Type (production): {model_type}")
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
        feature_names_list = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                               'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

        def _lo(p):
            p = float(np.clip(p, 1e-9, 1 - 1e-9))
            return float(np.log(p / (1 - p)))

        if model_type == 'RandomForest' and hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            total_imp = importances.sum() if importances.sum() > 0 else 1.0
            baseline_lo = _lo(pred_proba[1])
            feature_means = scaler.mean_
            for i, feature_name in enumerate(feature_names_list):
                original_val = list_data[i] if i < len(list_data) else 0
                imp = float(importances[i])
                # X is already a DataFrame — copy and perturb one column at a time
                X_pert = X.copy()
                X_pert.iloc[0, i] = feature_means[i]
                p_pert = float(model.predict_proba(X_pert)[0][1])
                direction_impact = baseline_lo - _lo(p_pert)
                feature_impacts.append({
                    'feature': feature_name,
                    'value': original_val,
                    'coefficient': round(imp, 4),
                    'impact': round(direction_impact, 3),
                    'normalized_impact': round(direction_impact, 3),
                    'relative_importance': round(imp / total_imp * 100, 1),
                })
        elif hasattr(model, 'coef_') and model.coef_.shape[0] == 1:
            coefficients = model.coef_[0]
            feature_means = scaler.mean_
            feature_scales = scaler.scale_
            total_abs_impact = sum(
                abs(((list_data[j] - feature_means[j]) / feature_scales[j]) * coefficients[j])
                for j in range(len(feature_names_list)) if j < len(list_data)
            ) or 1.0
            for i, feature_name in enumerate(feature_names_list):
                original_val = list_data[i] if i < len(list_data) else 0
                impact_score = ((original_val - feature_means[i]) / feature_scales[i]) * coefficients[i]
                denominator = abs(coefficients[i]) * feature_scales[i]
                normalized_impact = impact_score / denominator if denominator != 0 else 0.0
                feature_impacts.append({
                    'feature': feature_name,
                    'value': original_val,
                    'coefficient': round(float(coefficients[i]), 3),
                    'impact': round(impact_score, 3),
                    'normalized_impact': round(normalized_impact, 3),
                    'relative_importance': round(abs(impact_score) / total_abs_impact * 100, 1),
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
                try:
                    value = normalize_doctor_heart_cp(int(value))
                except (ValueError, TypeError):
                    missing_fields.append(field)
                    continue
            elif field == 'fbs':
                value = 1 if str(value).lower() in ['1', 'true', 'yes'] else 0
            elif field == 'restecg':
                value = int(value)  # 0,1,2
            elif field == 'exang':
                value = 1 if str(value).lower() in ['1', 'true', 'yes'] else 0
            elif field == 'slope':
                try:
                    value = normalize_doctor_heart_slope(int(value))
                except (ValueError, TypeError):
                    missing_fields.append(field)
                    continue
            elif field == 'ca':
                value = int(value)
                if value < 0 or value > 4:
                    missing_fields.append(field)
                    continue
            elif field == 'thal':
                value = int(value)
                if value < 0 or value > 3:
                    missing_fields.append(field)
                    continue

            try:
                list_data.append(float(value))
            except (ValueError, TypeError):
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

        model_path = 'heart_model.pkl'
        acc_path = 'heart_model_acc.txt'
        scaler_path = 'heart_model_scaler.pkl'
        if not all(os.path.exists(p) for p in (model_path, scaler_path, acc_path)):
            retrain_heart_model()
        else:
            try:
                with open(acc_path, 'r') as f:
                    _m = json.load(f)
                if int(_m.get('recipe_version', 0)) != DOCTOR_HEART_MODEL_RECIPE_VERSION:
                    retrain_heart_model()
            except (json.JSONDecodeError, OSError, TypeError, ValueError):
                retrain_heart_model()

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
        print(
            f"Doctor Confidence: healthy={healthy_prob * 100:.1f}%, "
            f"unhealthy={unhealthy_prob * 100:.1f}%"
        )
        
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
    categories = doc.exclude(category__isnull=True).exclude(category__exact='').values_list('category', flat=True).distinct().order_by('category')
    d = {'doc':doc, 'categories':categories}
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
        csv_path    = 'media/medical_dataset.csv'
        model_path  = 'patient_model.pkl'
        metrics_path = 'patient_model_metrics.json'

        df = pd.read_csv(csv_path)
        print("Patient data loaded, shape:", df.shape)

        X = df.drop('Result', axis=1)
        y = df['Result']
        feature_names = X.columns.tolist()
        print("Features used:", feature_names)
        print("Class distribution:")
        print(y.value_counts(normalize=True) * 100)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        print("Patient data split - train:", X_train.shape, "test:", X_test.shape)

        # ── Logistic Regression with GridSearchCV ──────────────────────────
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        lr_pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(max_iter=4000, class_weight='balanced',
                                        random_state=42, solver='lbfgs')),
        ])
        param_grid = {'clf__C': [0.1, 0.25, 0.5, 1.0, 2.0, 4.0]}
        print("Tuning patient Logistic Regression (5-fold stratified CV)...")
        grid = GridSearchCV(lr_pipe, param_grid, cv=cv, scoring='accuracy', n_jobs=1, refit=True)
        grid.fit(X_train, y_train)
        best_cv_lr = float(grid.best_score_)
        best_C     = float(grid.best_params_['clf__C'])
        print(f"LR  Best CV accuracy: {best_cv_lr*100:.2f}% (C={best_C})")

        lr_final_pipe = grid.best_estimator_
        scaler_lr     = lr_final_pipe.named_steps['scaler']
        lr_model      = lr_final_pipe.named_steps['clf']

        X_tr_s = scaler_lr.transform(X_train)
        X_te_s = scaler_lr.transform(X_test)
        y_pred_lr = lr_model.predict(X_te_s)
        acc_lr = float(accuracy_score(y_test, y_pred_lr))
        cm_lr  = confusion_matrix(y_test, y_pred_lr)
        rep_lr = classification_report(y_test, y_pred_lr)

        # ── Random Forest ──────────────────────────────────────────────────
        rf = RandomForestClassifier(
            n_estimators=300, max_depth=12, min_samples_leaf=2,
            class_weight='balanced', random_state=42, n_jobs=1,
        )
        print("Training patient Random Forest (300 trees)...")
        rf.fit(X_train, y_train)
        y_pred_rf = rf.predict(X_test)
        acc_rf  = float(accuracy_score(y_test, y_pred_rf))
        cm_rf   = confusion_matrix(y_test, y_pred_rf)
        rep_rf  = classification_report(y_test, y_pred_rf)

        # ── Terminal comparison ────────────────────────────────────────────
        print("=" * 60)
        print("PATIENT MODEL COMPARISON (80/20 stratified holdout)")
        print("=" * 60)
        print(f"\n--- Logistic Regression (best C={best_C}, CV={best_cv_lr*100:.2f}%) ---")
        print(f"Holdout accuracy: {acc_lr*100:.2f}%")
        print(f"TN={cm_lr[0][0]}, FP={cm_lr[0][1]}, FN={cm_lr[1][0]}, TP={cm_lr[1][1]}")
        print(rep_lr)

        print(f"\n--- Random Forest (300 trees) ---")
        print(f"Holdout accuracy: {acc_rf*100:.2f}%")
        print(f"TN={cm_rf[0][0]}, FP={cm_rf[0][1]}, FN={cm_rf[1][0]}, TP={cm_rf[1][1]}")
        print(rep_rf)

        f1_lr_0 = f1_score(y_test, y_pred_lr, pos_label=0)
        f1_lr_1 = f1_score(y_test, y_pred_lr, pos_label=1)
        f1_lr_m = f1_score(y_test, y_pred_lr, average='macro')
        f1_lr_w = f1_score(y_test, y_pred_lr, average='weighted')
        print(f"LR F1  healthy={f1_lr_0:.4f}  disease={f1_lr_1:.4f}  "
              f"macro={f1_lr_m:.4f}  weighted={f1_lr_w:.4f}")

        f1_rf_0 = f1_score(y_test, y_pred_rf, pos_label=0)
        f1_rf_1 = f1_score(y_test, y_pred_rf, pos_label=1)
        f1_rf_m = f1_score(y_test, y_pred_rf, average='macro')
        f1_rf_w = f1_score(y_test, y_pred_rf, average='weighted')
        print(f"RF F1  healthy={f1_rf_0:.4f}  disease={f1_rf_1:.4f}  "
              f"macro={f1_rf_m:.4f}  weighted={f1_rf_w:.4f}")

        # ── Winner selection ───────────────────────────────────────────────
        if acc_rf >= acc_lr:
            winner_model = rf
            winner_type  = 'RandomForest'
            winner_acc   = acc_rf
            loser_type   = 'LogisticRegression'
        else:
            winner_model = lr_model
            winner_type  = 'LogisticRegression'
            winner_acc   = acc_lr
            loser_type   = 'RandomForest'

        print(f"\n*** WINNER: {winner_type} (holdout={winner_acc*100:.2f}%) saved as production model ***")
        print(f"    ({loser_type} retained in terminal for analysis only)")
        print("=" * 60)

        # ── Save production bundle ─────────────────────────────────────────
        print("Saving patient model and metrics...")
        prod_bundle = {'model': winner_model, 'model_type': winner_type, 'scaler': scaler_lr}
        joblib.dump(prod_bundle, model_path)

        # ── Generate all evaluation plots ──────────────────────────────────
        try:
            generate_model_plots('patient', X_train, X_test, y_train, y_test,
                                  lr_model, rf, scaler_lr, feature_names, best_C=best_C)
        except Exception as plot_err:
            print(f"[WARN] Patient plot generation failed: {plot_err}")

        winner_cm  = cm_rf  if winner_type == 'RandomForest' else cm_lr
        winner_rep = rep_rf if winner_type == 'RandomForest' else rep_lr
        metrics = {
            'accuracy': winner_acc,
            'holdout_accuracy_lr': acc_lr,
            'holdout_accuracy_rf': acc_rf,
            'best_model_type': winner_type,
            'best_C': best_C,
            'cv_accuracy_lr': best_cv_lr,
            'feature_names': feature_names,
            'confusion_matrix': winner_cm.tolist(),
            'confusion_matrix_rf': cm_rf.tolist(),
            'confusion_matrix_lr': cm_lr.tolist(),
            'classification_report': winner_rep,
            'classification_report_rf': rep_rf,
            'classification_report_lr': rep_lr,
            'model_version': 'v2.0',
            'trained_on': str(datetime.now()),
            'f1_scores': {
                'lr_healthy': float(f1_lr_0), 'lr_disease': float(f1_lr_1),
                'lr_macro': float(f1_lr_m),   'lr_weighted': float(f1_lr_w),
                'rf_healthy': float(f1_rf_0), 'rf_disease': float(f1_rf_1),
                'rf_macro': float(f1_rf_m),   'rf_weighted': float(f1_rf_w),
            },
        }
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)

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
        
        def _load_patient_bundle():
            raw = joblib.load(model_path)
            m   = raw['model']
            mt  = raw.get('model_type', 'LogisticRegression')
            sc  = raw['scaler']
            return m, mt, sc

        model_type = 'LogisticRegression'
        try:
            model, model_type, scaler = _load_patient_bundle()
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
            feature_names = metrics.get('feature_names', [])
            accuracy = float(metrics.get('accuracy', 0.0)) * 100
            print(f"Loaded patient model ({model_type}) with accuracy: {accuracy:.2f}%")
        except Exception as e:
            print(f"Error loading patient model or metrics: {str(e)}. Retraining...")
            train_patient_model()
            model, model_type, scaler = _load_patient_bundle()
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
            feature_names = metrics.get('feature_names', [])
            accuracy = float(metrics.get('accuracy', 0.0)) * 100
            print(f"Retrained and loaded patient model ({model_type}) with accuracy: {accuracy:.2f}%")

        # Prepare input data for prediction by ensuring correct order and conversion to NumPy array
        input_data_list = [patient_input_data.get(feature, 0) for feature in feature_names]
        
        # Check for missing or invalid input (e.g., negative values where not expected)
        if any(val is None or (isinstance(val, (int, float)) and val < 0) for val in input_data_list):
            missing_or_invalid_features = [feature_names[i] for i, val in enumerate(input_data_list) if val is None or (isinstance(val, (int, float)) and val < 0)]
            raise ValueError(f"Missing or invalid input data for features: {missing_or_invalid_features}")

        X_predict_raw = np.array(input_data_list).reshape(1, -1)

        # RF uses raw features as a named DataFrame to avoid sklearn feature-name warnings
        if model_type == 'RandomForest':
            X_predict_input = pd.DataFrame(X_predict_raw, columns=feature_names)
        else:
            X_predict_input = scaler.transform(X_predict_raw)

        pred = model.predict(X_predict_input)
        pred_proba = model.predict_proba(X_predict_input)[0]
        print(f"Patient prediction ({model_type}):", pred[0])
        print("Patient prediction probabilities:", pred_proba)
        
        # Print current model metrics for reference
        print("\n" + "=" * 50)
        print("CURRENT PATIENT MODEL METRICS")
        print("=" * 50)
        print(f"Model Accuracy: {accuracy:.2f}%")
        print(f"Model Type (production): {model_type}")
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
        def _lo(p):
            p = float(np.clip(p, 1e-9, 1 - 1e-9))
            return float(np.log(p / (1 - p)))

        if model_type == 'RandomForest' and hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            total_imp = importances.sum() if importances.sum() > 0 else 1.0
            feature_means = scaler.mean_
            # Neutral baseline: all features at their training means
            X_neutral = pd.DataFrame([feature_means], columns=feature_names)
            neutral_lo = _lo(float(model.predict_proba(X_neutral)[0][1]))

            # Logically dependent features must be perturbed together, otherwise
            # we'd evaluate impossible records like "non-smoker with average smoking duration"
            FEATURE_GROUPS = [
                ['Smoke', 'Time_of_Smoking', 'Frequency_of_smoking'],
                ['Chest_Pain', 'Chest_Pain_Severity'],
                ['Short_Breath', 'Short_Breath_Duration'],
            ]
            group_of = {f: g for g in FEATURE_GROUPS for f in g}
            name_to_idx = {f: idx for idx, f in enumerate(feature_names)}

            for i, feature_name in enumerate(feature_names):
                original_val = patient_input_data.get(feature_name, 0)
                imp = float(importances[i])
                # Compare this patient's actual value(s) against the neutral/average patient
                X_single = X_neutral.copy()
                group = group_of.get(feature_name, [feature_name])
                for gf in group:
                    if gf in name_to_idx:
                        X_single.iloc[0, name_to_idx[gf]] = patient_input_data.get(gf, 0)
                p_single = float(model.predict_proba(X_single)[0][1])
                direction_impact = _lo(p_single) - neutral_lo
                feature_impacts.append({
                    'feature': feature_name,
                    'value': original_val,
                    'coefficient': round(imp, 4),
                    'impact': round(direction_impact, 3),
                    'normalized_impact': round(direction_impact, 3),
                    'relative_importance': round(imp / total_imp * 100, 1),
                })
        elif hasattr(model, 'coef_') and model.coef_.shape[0] == 1:
            coefficients  = model.coef_[0]
            feature_means = scaler.mean_
            feature_scales = scaler.scale_
            total_abs = sum(
                abs(((patient_input_data.get(feature_names[j], 0) - feature_means[j])
                     / feature_scales[j]) * coefficients[j])
                for j in range(len(feature_names))
            ) or 1.0
            for i, feature_name in enumerate(feature_names):
                original_val = patient_input_data.get(feature_name, 0)
                impact_score = ((original_val - feature_means[i]) / feature_scales[i]) * coefficients[i]
                denom = abs(coefficients[i]) * feature_scales[i]
                normalized_impact = impact_score / denom if denom != 0 else 0.0
                feature_impacts.append({
                    'feature': feature_name,
                    'value': original_val,
                    'coefficient': round(float(coefficients[i]), 3),
                    'impact': round(impact_score, 3),
                    'normalized_impact': round(normalized_impact, 3),
                    'relative_importance': round(abs(impact_score) / total_abs * 100, 1),
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
        print(
            f"Patient Confidence: healthy={pred_proba[0] * 100:.1f}%, "
            f"unhealthy={pred_proba[1] * 100:.1f}%"
        )
        
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
            'feature_impacts': feature_impacts,
            'is_patient': True,
            'pred_value': int(pred_value),
            'unhealthy_prob_pct': round(float(pred_proba[1]) * 100, 1),
        })
        
    return render(request, 'add_heartdetail_patient.html', {'patient_name': patient_name, 'patient_age': patient_age})


def nearby_hospitals(request):
    import math
    try:
        lat = float(request.GET.get('lat', 0))
        lng = float(request.GET.get('lng', 0))
        radius = float(request.GET.get('radius', 5))
    except (TypeError, ValueError):
        return JsonResponse({'error': 'Invalid coordinates'}, status=400)

    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Machine_Learning', 'hospitals_nepal.json')
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            all_hospitals = json.load(f)
    except FileNotFoundError:
        return JsonResponse({'error': 'Hospital data not found'}, status=500)

    def haversine(lat1, lon1, lat2, lon2):
        R = 6371
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    EXCLUDE = ['beauty', 'parlour', 'parlor', 'pathology', 'laboratory', 'lab ', 'diagnostic',
               'optical', 'pharmacy', 'veterinary', 'animal', 'pet clinic', 'pet hospital',
               'ayurvedic', 'herbal', 'tibetan', 'tibetian', 'traditional medicine', 'medicine center',
               'dental', 'dentist', 'skin care', 'spa ', 'massage', 'blind', 'mata,', 'temple', 'mandir',
               'eye center', 'eye centre', 'eye hospital', 'vision eye', 'trauma center', 'trauma centre',
               'jgo']

    def is_medical(name):
        n = name.lower()
        return not any(kw in n for kw in EXCLUDE)

    nearby = []
    for h in all_hospitals:
        if not is_medical(h['name']):
            continue
        d = haversine(lat, lng, h['lat'], h['lon'])
        if d <= radius:
            nearby.append({**h, 'distance': round(d, 3)})

    nearby.sort(key=lambda x: x['distance'])
    return JsonResponse({'hospitals': nearby[:20]})
