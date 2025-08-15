#!/usr/bin/env python3
"""
Quick version of Cardiovascular Fitness Model Training
Reduced complexity for faster training
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import json
from datetime import datetime

np.random.seed(42)

print("="*60)
print("CARDIOVASCULAR FITNESS MODEL - QUICK TRAINING")
print("="*60)

# Generate simplified training data
n_samples = 2000
print(f"\nGenerating {n_samples} training samples...")

data = []
for i in range(n_samples):
    age = np.random.randint(18, 80)
    
    # Base fitness (0-100)
    fitness = 70 - (age - 40) * 0.5 + np.random.normal(0, 10)
    fitness = np.clip(fitness, 10, 95)
    
    # Resting heart rate (inversely correlated with fitness)
    rhr = 85 - fitness * 0.35 + np.random.normal(0, 3)
    rhr = np.clip(rhr, 40, 95)
    
    # Heart rate recovery 1 minute
    if fitness > 70:
        hrr_1min = 30 + (fitness - 70) * 0.5 + np.random.normal(0, 3)
    elif fitness > 40:
        hrr_1min = 20 + (fitness - 40) * 0.33 + np.random.normal(0, 2)
    else:
        hrr_1min = 12 + (fitness - 20) * 0.4 + np.random.normal(0, 2)
    hrr_1min = np.clip(hrr_1min, 5, 50)
    
    # HRV (RMSSD)
    rmssd = 20 + fitness * 0.6 + np.random.normal(0, 5)
    rmssd = np.clip(rmssd, 10, 100)
    
    # VO2max estimate
    vo2max = 25 + fitness * 0.3 + np.random.normal(0, 3)
    vo2max = np.clip(vo2max, 15, 75)
    
    # Cardiovascular age
    cv_age = age + (50 - fitness) * 0.3 + np.random.normal(0, 3)
    cv_age = np.clip(cv_age, 18, 90)
    
    data.append({
        'age': age,
        'resting_hr': rhr,
        'hrr_1min': hrr_1min,
        'rmssd': rmssd,
        'fitness_level': fitness,
        'vo2max': vo2max,
        'cardiovascular_age': cv_age
    })

df = pd.DataFrame(data)
print(f"Generated dataset with {len(df)} samples")

# Prepare features
feature_cols = ['age', 'resting_hr', 'hrr_1min', 'rmssd']
X = df[feature_cols].values

# Train fitness level model
print("\n" + "="*60)
print("Training Fitness Level Model")
print("="*60)

y_fitness = df['fitness_level'].values
X_train, X_test, y_train, y_test = train_test_split(X, y_fitness, test_size=0.2, random_state=42)

scaler_fitness = StandardScaler()
X_train_scaled = scaler_fitness.fit_transform(X_train)
X_test_scaled = scaler_fitness.transform(X_test)

rf_fitness = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
rf_fitness.fit(X_train_scaled, y_train)

y_pred = rf_fitness.predict(X_test_scaled)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Fitness Level Model Performance:")
print(f"  MAE: {mae:.2f}")
print(f"  R²: {r2:.3f}")

# Train VO2max model
print("\n" + "="*60)
print("Training VO2max Model")
print("="*60)

y_vo2max = df['vo2max'].values
X_train, X_test, y_train, y_test = train_test_split(X, y_vo2max, test_size=0.2, random_state=42)

scaler_vo2max = StandardScaler()
X_train_scaled = scaler_vo2max.fit_transform(X_train)
X_test_scaled = scaler_vo2max.transform(X_test)

rf_vo2max = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
rf_vo2max.fit(X_train_scaled, y_train)

y_pred = rf_vo2max.predict(X_test_scaled)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"VO2max Model Performance:")
print(f"  MAE: {mae:.2f} ml/kg/min")
print(f"  R²: {r2:.3f}")

# Train cardiovascular age model
print("\n" + "="*60)
print("Training Cardiovascular Age Model")
print("="*60)

y_cv_age = df['cardiovascular_age'].values
X_train, X_test, y_train, y_test = train_test_split(X, y_cv_age, test_size=0.2, random_state=42)

scaler_cv_age = StandardScaler()
X_train_scaled = scaler_cv_age.fit_transform(X_train)
X_test_scaled = scaler_cv_age.transform(X_test)

rf_cv_age = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
rf_cv_age.fit(X_train_scaled, y_train)

y_pred = rf_cv_age.predict(X_test_scaled)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Cardiovascular Age Model Performance:")
print(f"  MAE: {mae:.1f} years")
print(f"  R²: {r2:.3f}")

# Feature importance
print("\n" + "="*60)
print("Feature Importance Analysis")
print("="*60)

importance = rf_fitness.feature_importances_
for feature, imp in zip(feature_cols, importance):
    print(f"  {feature}: {imp:.3f}")

# Save models
print("\n" + "="*60)
print("Saving Models")
print("="*60)

# Save models and scalers
joblib.dump((rf_fitness, scaler_fitness), 'cardiovascular_fitness_model.pkl')
joblib.dump((rf_vo2max, scaler_vo2max), 'cardiovascular_vo2max_model.pkl')
joblib.dump((rf_cv_age, scaler_cv_age), 'cardiovascular_age_model.pkl')

print("Models saved successfully!")

# Save configuration
config = {
    'feature_cols': feature_cols,
    'model_type': 'RandomForestRegressor',
    'n_samples': n_samples,
    'timestamp': datetime.now().isoformat(),
    'performance': {
        'fitness_r2': float(r2_score(df['fitness_level'], rf_fitness.predict(scaler_fitness.transform(df[feature_cols])))),
        'vo2max_r2': float(r2_score(df['vo2max'], rf_vo2max.predict(scaler_vo2max.transform(df[feature_cols])))),
        'cv_age_r2': float(r2_score(df['cardiovascular_age'], rf_cv_age.predict(scaler_cv_age.transform(df[feature_cols]))))
    }
}

with open('cardiovascular_model_config.json', 'w') as f:
    json.dump(config, f, indent=2)

print("Configuration saved!")

print("\n" + "="*60)
print("TRAINING COMPLETE")
print("="*60)