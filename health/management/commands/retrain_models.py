from django.core.management.base import BaseCommand
from health.views import retrain_heart_model, train_patient_model
import os

class Command(BaseCommand):
    help = 'Retrain both heart disease and patient models with detailed metrics printing'

    def add_arguments(self, parser):
        parser.add_argument(
            '--heart-only',
            action='store_true',
            help='Retrain only the heart disease model',
        )
        parser.add_argument(
            '--patient-only',
            action='store_true',
            help='Retrain only the patient model',
        )

    def handle(self, *args, **options):
        self.stdout.write(
            self.style.SUCCESS('Starting model retraining with detailed metrics...')
        )
        
        try:
            if options['patient_only']:
                self.stdout.write('Retraining patient model only...')
                train_patient_model()
                self.stdout.write(
                    self.style.SUCCESS('Patient model retraining completed!')
                )
            elif options['heart_only']:
                self.stdout.write('Retraining heart disease model only...')
                retrain_heart_model()
                self.stdout.write(
                    self.style.SUCCESS('Heart disease model retraining completed!')
                )
            else:
                self.stdout.write('Retraining both models...')
                
                # Retrain heart disease model
                self.stdout.write('Retraining heart disease model...')
                retrain_heart_model()
                
                # Retrain patient model
                self.stdout.write('Retraining patient model...')
                train_patient_model()
                
                self.stdout.write(
                    self.style.SUCCESS('Both models retraining completed!')
                )
                
        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f'Error during model retraining: {str(e)}')
            )
            raise 