"""
Synthetic dataset generator for the patient-facing heart risk model.

Generates patients with realistic, correlated vital signs/lifestyle
fields, then derives Result via a two-stage process: a "core" clinical
risk latent (age, BP, cholesterol, diabetes, smoking, family history,
BMI/waist ratio, resting HR, lifestyle) drives both the probability of
reporting symptoms (chest pain, breathlessness, palpitations, swelling,
dizziness) AND the final disease probability, with substantial noise at
each stage. This keeps any single column (notably Age, which dominated
the v1 dataset) from being a giveaway, while remaining clinically
plausible and learnable by the production LR/RF tournament.

Run: python Machine_Learning/generate_patient_dataset.py
Writes: media/medical_dataset.csv
"""
import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
N = 12000


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def clip(arr, lo, hi):
    return np.clip(arr, lo, hi)


def main():
    n = N

    # ---- Demographics ----------------------------------------------------
    age = RNG.integers(18, 91, size=n)
    gender = RNG.integers(0, 2, size=n)  # 0=Male, 1=Female

    height = np.where(
        gender == 0,
        RNG.normal(170, 7, n),
        RNG.normal(158, 6, n),
    )
    height = clip(height, 140, 205).round(0)

    base_bmi = RNG.normal(26, 5, n)
    base_bmi = clip(base_bmi, 16, 48)
    weight = base_bmi * (height / 100) ** 2
    weight = clip(weight, 35, 220).round(1)
    bmi = (weight / ((height / 100) ** 2)).round(2)

    # ---- Smoking ------------------------------------------------------
    smoke_prob = clip(0.12 + 0.15 * (age < 60) + 0.05 * (gender == 0), 0.05, 0.4)
    smoke = (RNG.random(n) < smoke_prob).astype(int)
    time_of_smoking = np.where(
        smoke == 1, RNG.integers(1, np.maximum(age - 15, 2), size=n), 0
    )
    frequency_of_smoking = np.where(smoke == 1, RNG.integers(1, 41, size=n), 0)

    # ---- Vitals correlated with age/bmi/lifestyle ----------------------
    stress = RNG.integers(0, 4, size=n)
    exercise = RNG.integers(0, 4, size=n)
    fatty_food = RNG.integers(0, 4, size=n)
    alcohol = RNG.integers(0, 4, size=n)
    sleep_quality = RNG.integers(0, 3, size=n)  # 0=good,1=fair,2=poor

    systolic_bp = (
        100
        + 0.4 * (age - 18)
        + 0.6 * (bmi - 22)
        + 2.0 * stress
        + 3.0 * smoke
        + RNG.normal(0, 9, n)
    )
    systolic_bp = clip(systolic_bp, 90, 200).round(0)

    resting_hr = (
        68
        + 0.15 * (bmi - 22)
        + 4.0 * smoke
        + 2.0 * stress
        - 2.0 * exercise
        + RNG.normal(0, 8, n)
    )
    resting_hr = clip(resting_hr, 48, 130).round(0)

    waist_height_ratio = 0.42 + 0.012 * (bmi - 22) + RNG.normal(0, 0.03, n)
    waist_height_ratio = clip(waist_height_ratio, 0.35, 0.78).round(3)

    # ---- Ternary clinical flags (0=No, 1=Yes, 2=Not sure) ---------------
    def ternary(yes_prob, unsure_rate=0.08):
        u = RNG.random(n)
        yes = (RNG.random(n) < yes_prob).astype(int)
        out = np.where(u < unsure_rate, 2, yes)
        return out

    hbp_prob = clip(0.05 + 0.006 * (age - 18) + 0.01 * (systolic_bp - 120), 0.03, 0.85)
    high_blood_pressure = ternary(hbp_prob)

    diabetes_prob = clip(0.03 + 0.005 * (age - 18) + 0.015 * (bmi - 22), 0.02, 0.7)
    diabetes = ternary(diabetes_prob)

    chol_prob = clip(0.05 + 0.005 * (age - 18) + 0.02 * fatty_food + 0.01 * (bmi - 22), 0.03, 0.75)
    high_cholesterol = ternary(chol_prob)

    fh_prob = np.full(n, 0.28)
    family_history = ternary(fh_prob, unsure_rate=0.10)
    # Earlier onset (lower age) is a stronger signal; 0 = not applicable
    family_history_age = np.where(
        family_history == 1, RNG.integers(28, 86, size=n), 0
    )

    # ---- Map ternary flags to a 0/0.4/1 risk value for the core latent --
    def risk_val(arr):
        return np.select([arr == 0, arr == 1, arr == 2], [0.0, 1.0, 0.4])

    hbp_val = risk_val(high_blood_pressure)
    dia_val = risk_val(diabetes)
    chol_val = risk_val(high_cholesterol)
    fh_val = risk_val(family_history)

    fh_onset_bonus = np.where(
        family_history == 1, clip((60 - family_history_age) / 40.0, -0.5, 1.0), 0.0
    )

    # ---- Core clinical risk latent (no symptoms yet) ---------------------
    core = (
        0.85 * ((age - 45) / 15)
        + 0.15 * (gender == 0)
        + 0.55 * smoke
        + 0.015 * time_of_smoking
        + 0.02 * frequency_of_smoking
        + 0.55 * hbp_val + 0.02 * (systolic_bp - 120)
        + 0.50 * dia_val
        + 0.45 * chol_val
        + 0.35 * fh_val + 0.40 * fh_onset_bonus
        + 0.35 * ((bmi - 25) / 6)
        + 0.25 * ((waist_height_ratio - 0.5) / 0.1)
        + 0.25 * ((resting_hr - 70) / 15)
        + 0.12 * alcohol
        + 0.18 * sleep_quality
        + 0.12 * stress
        - 0.18 * exercise
        + 0.12 * fatty_food
        + 0.25 * ((smoke == 1) & (age > 50))
        + 0.20 * ((hbp_val > 0.5) & (dia_val > 0.5))
        + 0.20 * ((bmi > 30) & (exercise == 0))
    )

    # ---- Symptoms: noisy manifestations of the core latent ---------------
    def freq_from_latent(latent, n_levels, spread=1.0, noise_sd=1.0):
        z = latent / spread + RNG.normal(0, noise_sd, n)
        # split z into n_levels buckets via quantile-free fixed thresholds
        thresholds = np.linspace(-1.2, 1.6, n_levels - 1)
        levels = np.zeros(n, dtype=int)
        for t in thresholds:
            levels += (z > t).astype(int)
        return clip(levels, 0, n_levels - 1)

    chest_pain = freq_from_latent(core, 4, spread=1.3, noise_sd=1.1)
    chest_pain_severity = freq_from_latent(core, 5, spread=1.4, noise_sd=1.2)
    short_breath = freq_from_latent(core, 4, spread=1.3, noise_sd=1.1)
    short_breath_duration = freq_from_latent(core, 5, spread=1.5, noise_sd=1.2)
    palpitations = freq_from_latent(core, 4, spread=1.5, noise_sd=1.3)
    dizziness = freq_from_latent(core, 4, spread=1.6, noise_sd=1.3)
    swelling_legs = (sigmoid(0.8 * core - 1.0) + RNG.normal(0, 0.25, n) > 0.5).astype(int)

    # ---- Final risk = core + modest symptom contribution + noise ---------
    symptom_term = (
        0.15 * chest_pain + 0.08 * chest_pain_severity
        + 0.13 * short_breath + 0.05 * short_breath_duration
        + 0.12 * palpitations + 0.10 * dizziness + 0.35 * swelling_legs
    )
    final_latent = core + symptom_term + RNG.normal(0, 0.55, n)

    prob = sigmoid(0.85 * (final_latent - np.median(final_latent)))
    result = (RNG.random(n) < prob).astype(int)

    df = pd.DataFrame({
        "Age": age,
        "Gender": gender,
        "Height": height.astype(int),
        "Weight": weight,
        "BMI": bmi,
        "Smoke": smoke,
        "Time_of_Smoking": time_of_smoking,
        "Frequency_of_smoking": frequency_of_smoking,
        "High_Blood_Pressure": high_blood_pressure,
        "Diabetes": diabetes,
        "High_Cholesterol": high_cholesterol,
        "Family_History": family_history,
        "Chest_Pain": chest_pain,
        "Chest_Pain_Severity": chest_pain_severity,
        "Short_Breath": short_breath,
        "Short_Breath_Duration": short_breath_duration,
        "Exercise": exercise,
        "Fatty_Food": fatty_food,
        "Stress": stress,
        "Resting_Heart_Rate": resting_hr.astype(int),
        "Systolic_BP": systolic_bp.astype(int),
        "Waist_Height_Ratio": waist_height_ratio,
        "Alcohol_Consumption": alcohol,
        "Sleep_Quality": sleep_quality,
        "Palpitations": palpitations,
        "Swelling_Legs": swelling_legs,
        "Dizziness": dizziness,
        "Family_History_Age": family_history_age,
        "Result": result,
    })

    out_path = "media/medical_dataset.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path}")
    print("Class balance:\n", df["Result"].value_counts(normalize=True))
    print("\nCorrelation with Result:\n", df.corr(numeric_only=True)["Result"].sort_values(ascending=False))


if __name__ == "__main__":
    main()
