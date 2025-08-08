from django.db import models
from django.contrib.auth.models import User

# Create your models here.
from .choices import DOCTOR_STATUS

class Patient(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, null=True)
    contact = models.CharField(max_length=100, null=True)
    address = models.CharField(max_length=100, null=True)
    dob = models.DateField(null=True)
    image = models.FileField(null=True)

    def __str__(self):
        return self.user.username

class Doctor(models.Model):
    status = models.IntegerField(DOCTOR_STATUS, null=True)
    user = models.OneToOneField(User, on_delete=models.CASCADE, null=True)
    contact = models.CharField(max_length=100, null=True)
    address = models.CharField(max_length=100, null=True)
    category = models.CharField(max_length=100, null=True)
    doj = models.DateField(null=True)
    dob = models.DateField(null=True)
    image = models.FileField(null=True)

    def __str__(self):
        return self.user.username

class Admin_Helath_CSV(models.Model):
    name = models.CharField(max_length=100, null=True)
    csv_file = models.FileField(null=True, blank=True)

    def __str__(self):
        return self.name

class Search_Data(models.Model):
    patient = models.ForeignKey(Patient, on_delete=models.CASCADE, null=True, blank=True)
    doctor = models.ForeignKey(Doctor, on_delete=models.CASCADE, null=True, blank=True)
    prediction_accuracy = models.CharField(max_length=100,null=True,blank=True)
    result = models.CharField(max_length=100,null=True,blank=True)
    values_list = models.CharField(max_length=100,null=True,blank=True)
    feature_impacts = models.TextField(null=True, blank=True)
    created = models.DateTimeField(auto_now=True,null=True)

    def __str__(self):
        if self.patient:
            return f"{self.patient.user.username} - {self.created}"
        elif self.doctor:
            return f"{self.doctor.user.username} - {self.created}"
        return f"Unknown - {self.created}"

class Feedback(models.Model):
    name = models.CharField(max_length=200, null=True)
    email = models.EmailField(max_length=200, null=True)
    contact = models.CharField(max_length=100, null=True, blank=True)
    subject = models.CharField(max_length=500, null=True)
    messages = models.TextField(null=True)
    date = models.DateField(auto_now=True)
    is_read = models.BooleanField(default=False)

    def __str__(self):
        return self.name if self.name else f"Feedback #{self.id}"

class PredictionHistory(models.Model):
    search_data = models.ForeignKey(Search_Data, on_delete=models.CASCADE)
    prediction_accuracy = models.CharField(max_length=100, null=True, blank=True)
    result = models.CharField(max_length=100, null=True, blank=True)
    values_list = models.CharField(max_length=100, null=True, blank=True)
    feature_impacts = models.TextField(null=True, blank=True)
    created = models.DateTimeField(auto_now_add=True)
    updated = models.DateTimeField(auto_now=True)

    def __str__(self):
        if self.search_data.patient:
            return f"{self.search_data.patient.user.username} - {self.created}"
        elif self.search_data.doctor:
            return f"{self.search_data.doctor.user.username} - {self.created}"
        return f"Unknown - {self.created}"