3.1.1.1 Functional Requirements

1. User Management
   i. User Registration and Authentication
      a. Patients can register and create accounts
      b. Only administrators can register doctor accounts
      c. Admin account for system management
      d. Secure login system for all users (patients, doctors, admin)

2. Health Parameter Management
   i. Patient Data Input
      a. Simplified health assessment form with:
         • Age and Sex
         • Height and Weight
         • Smoking status
         • Medical history (hypertension, diabetes, high cholesterol)
         • Family history of heart disease
         • Current symptoms (chest pain, shortness of breath)
         • Lifestyle factors (exercise frequency, diet habits, stress levels)
   ii. Doctor Data Input
       a. Detailed clinical parameters:
          • Age
          • Sex
          • Chest Pain Type (cp)
          • Resting Blood Pressure (trestbps)
          • Cholesterol (chol)
          • Fasting Blood Sugar (fbs)
          • Resting ECG (restecg)
          • Maximum Heart Rate (thalach)
          • Exercise Induced Angina (exang)
          • ST Depression (oldpeak)
          • Slope of Peak Exercise ST Segment (slope)
          • Number of Major Vessels (ca)
          • Thalassemia (thal)
   iii. Prediction Model
        a. Uses Gradient Boosting Classifier
        b. Provides binary classification (Healthy/Unhealthy)
        c. Shows prediction confidence percentage

3. Prediction History Management
   i. For Patients:
      a. View their own prediction history
      b. Access detailed prediction results
   ii. For Doctors:
      a. Make predictions for patients
      b. View predictions they've made
   iii. For Admin:
      a. View all predictions in the system
      b. Access system-wide prediction data

4. Role-Based Access Control
   i. Patient Features:
      a. Input health parameters
      b. View personal predictions
      c. Access own profile
   ii. Doctor Features:
      a. Make predictions for patients
      b. View their prediction history
   iii. Admin Features:
      a. System-wide access
      b. View all predictions
      c. Manage user accounts

5. Data Visualization
   i. Display prediction results
      a. Clear indication of prediction outcome (Healthy/Unhealthy)
      b. Display of prediction confidence percentage
   ii. Tabular display of prediction history

6. Profile Management
   i. Users can view their profiles
   ii. Basic profile information management
   iii. Role-specific access restrictions

7. System Administration
   i. Monitor prediction activities
   ii. Basic user account management
   iii. System maintenance capabilities 