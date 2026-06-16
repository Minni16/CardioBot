"""
Cleans media/medical_dataset.csv by removing rows with impossible
anthropometric values and fixing logical inconsistencies.

Run once from the project root:
    python clean_patient_dataset.py
"""
import pandas as pd
import os

INPUT_PATH = 'media/medical_dataset.csv'


def main():
    if not os.path.exists(INPUT_PATH):
        print(f"ERROR: {INPUT_PATH} not found.")
        return

    df = pd.read_csv(INPUT_PATH)
    print(f"Original shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print("Original class distribution:")
    print(df['Result'].value_counts().to_string())
    print(f"Original BMI range:    {df['BMI'].min():.1f} – {df['BMI'].max():.1f}")
    print(f"Original Height range: {df['Height'].min()} – {df['Height'].max()} cm")
    print(f"Original Weight range: {df['Weight'].min()} – {df['Weight'].max()} kg")

    # --- Step 1: Anthropometric filters ---
    before = len(df)

    df = df[(df['Height'] >= 140) & (df['Height'] <= 220)]
    after_height = len(df)
    print(f"\nRemoved {before - after_height} rows with Height outside 140–220 cm")

    df = df[(df['Weight'] >= 40) & (df['Weight'] <= 180)]
    after_weight = len(df)
    print(f"Removed {after_height - after_weight} rows with Weight outside 40–180 kg")

    # Recompute BMI from actual height/weight
    df = df.copy()
    df['BMI'] = (df['Weight'] / ((df['Height'] / 100) ** 2)).round(2)

    df = df[(df['BMI'] >= 15) & (df['BMI'] <= 60)]
    after_bmi = len(df)
    print(f"Removed {after_weight - after_bmi} rows with BMI outside 15–60 after recomputation")

    # --- Step 2: Logical consistency fixes ---
    # Non-smokers can't have smoking duration or frequency
    inconsistent_smoke = ((df['Smoke'] == 0) & (df['Time_of_Smoking'] > 0)).sum()
    df.loc[df['Smoke'] == 0, 'Time_of_Smoking'] = 0
    df.loc[df['Smoke'] == 0, 'Frequency_of_smoking'] = 0
    print(f"\nFixed {inconsistent_smoke} rows where Smoke=0 but Time_of_Smoking>0")

    # No chest pain → severity must be 0
    inconsistent_cp = ((df['Chest_Pain'] == 0) & (df['Chest_Pain_Severity'] > 0)).sum()
    df.loc[df['Chest_Pain'] == 0, 'Chest_Pain_Severity'] = 0
    print(f"Fixed {inconsistent_cp} rows where Chest_Pain=0 but Chest_Pain_Severity>0")

    # No shortness of breath → duration must be 0
    inconsistent_sb = ((df['Short_Breath'] == 0) & (df['Short_Breath_Duration'] > 0)).sum()
    df.loc[df['Short_Breath'] == 0, 'Short_Breath_Duration'] = 0
    print(f"Fixed {inconsistent_sb} rows where Short_Breath=0 but Short_Breath_Duration>0")

    # --- Step 3: Reset index and save ---
    df = df.reset_index(drop=True)
    df.to_csv(INPUT_PATH, index=False)

    print(f"\n{'='*50}")
    print(f"CLEANED dataset: {len(df)} rows  (removed {before - len(df)} total)")
    print("Class distribution:")
    print(df['Result'].value_counts().to_string())
    print(f"BMI range:    {df['BMI'].min():.1f} – {df['BMI'].max():.1f}")
    print(f"Height range: {df['Height'].min()} – {df['Height'].max()} cm")
    print(f"Weight range: {df['Weight'].min()} – {df['Weight'].max()} kg")
    print(f"\nSaved to {INPUT_PATH}")
    print("Delete patient_model.pkl and patient_model_metrics.json to trigger retrain.")


if __name__ == '__main__':
    main()
