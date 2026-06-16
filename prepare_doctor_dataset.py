"""
Preprocesses Machine_Learning/heart_disease_uci.csv (UCI combined, 920 rows)
into the format expected by the doctor model and saves it as media/heart.csv.

Run once from the project root:
    python prepare_doctor_dataset.py
"""
import pandas as pd
import numpy as np
import os

INPUT_PATH = 'Machine_Learning/heart_disease_uci.csv'
OUTPUT_PATH = 'media/heart.csv'


def main():
    if not os.path.exists(INPUT_PATH):
        print(f"ERROR: {INPUT_PATH} not found. Place the UCI combined CSV there first.")
        return

    df = pd.read_csv(INPUT_PATH)
    print(f"Loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    # --- Rename ---
    df = df.rename(columns={'thalch': 'thalach'})

    # --- Drop metadata columns ---
    df = df.drop(columns=['id', 'dataset'], errors='ignore')

    # --- Encode categorical text columns ---
    df['sex'] = df['sex'].map({'Male': 1, 'Female': 0})

    cp_map = {
        'typical angina': 0,
        'atypical angina': 1,
        'non-anginal': 2,
        'asymptomatic': 3,
    }
    df['cp'] = df['cp'].map(cp_map)

    thal_map = {
        'fixed defect': 1,
        'normal': 2,
        'reversable defect': 3,
    }
    df['thal'] = df['thal'].map(thal_map)

    slope_map = {
        'upsloping': 0,
        'flat': 1,
        'downsloping': 2,
    }
    df['slope'] = df['slope'].map(slope_map)

    df['fbs'] = df['fbs'].map({True: 1, False: 0, 1: 1, 0: 0})
    df['exang'] = df['exang'].map({True: 1, False: 0, 1: 1, 0: 0})

    restecg_map = {
        'normal': 0,
        'st-t abnormality': 1,
        'lv hypertrophy': 2,
    }
    df['restecg'] = df['restecg'].map(restecg_map)

    # --- Fix chol=0 (coded missing in original sources) ---
    df.loc[df['chol'] == 0, 'chol'] = np.nan

    # --- Impute missing values ---
    # Numerical: median
    for col in ['trestbps', 'chol', 'thalach', 'oldpeak']:
        median_val = df[col].median()
        n_missing = df[col].isna().sum()
        if n_missing > 0:
            df[col] = df[col].fillna(median_val)
            print(f"  Imputed {n_missing} missing '{col}' values with median={median_val:.1f}")

    # Categorical: mode
    for col in ['fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']:
        mode_val = df[col].mode()[0]
        n_missing = df[col].isna().sum()
        if n_missing > 0:
            df[col] = df[col].fillna(mode_val)
            print(f"  Imputed {n_missing} missing '{col}' values with mode={mode_val}")

    # --- Binarize target: 0=no disease, 1+=disease -> 1 ---
    df['target'] = (df['num'] > 0).astype(int)
    df = df.drop(columns=['num'])

    # --- Cast to correct types ---
    int_cols = ['age', 'sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal', 'target']
    for col in int_cols:
        df[col] = df[col].astype(int)

    float_cols = ['trestbps', 'chol', 'thalach', 'oldpeak']
    for col in float_cols:
        df[col] = df[col].round(1)

    # --- Final column order (matches existing heart.csv) ---
    col_order = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
    df = df[col_order]

    # --- Save ---
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"\nSaved {len(df)} rows to {OUTPUT_PATH}")
    print("Class distribution:")
    print(df['target'].value_counts().to_string())
    print(f"\nAge range:     {df['age'].min()} – {df['age'].max()}")
    print(f"Chol range:    {df['chol'].min()} – {df['chol'].max()}")
    print(f"Thalach range: {df['thalach'].min()} – {df['thalach'].max()}")
    print("\nDone. Remember to bump DOCTOR_HEART_MODEL_RECIPE_VERSION in health/views.py.")


if __name__ == '__main__':
    main()
