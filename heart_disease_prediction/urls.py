from django.urls import path
from . import views

urlpatterns = [
    path('predict_heart_disease_patient/', views.predict_heart_disease_patient, name='predict_heart_disease_patient'),
    path('train_normal_user_model/', views.train_normal_user_model, name='train_normal_user_model'),
] 