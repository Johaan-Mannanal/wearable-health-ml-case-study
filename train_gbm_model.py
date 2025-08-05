#!/usr/bin/env python3
"""
Train the enhanced GBM model for health risk assessment
Adapted for Apple Watch Series 10 (no blood pressure required)
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.pipeline import Pipeline
import joblib
import json
from datetime import datetime

print("Starting GBM model training for Apple Watch health risk assessment...")

# Step 1: Generate synthetic training data (10000 samples - smaller than original)
print("\n1. Generating synthetic training data...")
np.random.seed(42)
n_samples = 10000

def generate_health_data(n_samples):
    # Low risk profiles (60%)
    n_low_risk = int(n_samples * 0.6)
    low_risk = pd.DataFrame({
        'average_heart_rate': np.random.normal(70, 8, n_low_risk),
        'hrv_mean': np.random.normal(50, 15, n_low_risk),
        'respiratory_rate': np.random.normal(14, 2, n_low_risk),
        'activity_level': np.random.gamma(3, 100, n_low_risk),  # steps per hour
        'sleep_quality': np.random.beta(7, 3, n_low_risk),  # 0-1 scale
        'stress_indicator': np.random.beta(2, 5, n_low_risk)  # derived from HRV
    })
    low_risk['risk_level'] = 0
    
    # High risk profiles (40%)
    n_high_risk = n_samples - n_low_risk
    high_risk = pd.DataFrame({
        'average_heart_rate': np.concatenate([
            np.random.normal(90, 12, n_high_risk//2),  # elevated HR
            np.random.normal(55, 8, n_high_risk//2)    # low HR
        ]),
        'hrv_mean': np.random.normal(30, 10, n_high_risk),  # lower HRV
        'respiratory_rate': np.random.normal(18, 3, n_high_risk),  # elevated
        'activity_level': np.random.gamma(1, 50, n_high_risk),  # less active
        'sleep_quality': np.random.beta(3, 7, n_high_risk),  # poor sleep
        'stress_indicator': np.random.beta(5, 2, n_high_risk)  # high stress
    })
    high_risk['risk_level'] = 1
    
    # Combine data
    data = pd.concat([low_risk, high_risk], ignore_index=True)
    
    # Apply physiological constraints
    data['average_heart_rate'] = np.clip(data['average_heart_rate'], 40, 120)
    data['hrv_mean'] = np.clip(data['hrv_mean'], 10, 100)
    data['respiratory_rate'] = np.clip(data['respiratory_rate'], 8, 25)
    data['activity_level'] = np.clip(data['activity_level'], 0, 1000)
    data['sleep_quality'] = np.clip(data['sleep_quality'], 0, 1)
    data['stress_indicator'] = np.clip(data['stress_indicator'], 0, 1)
    
    # Add derived features
    data['hr_hrv_ratio'] = data['average_heart_rate'] / (data['hrv_mean'] + 1)
    data['recovery_score'] = data['sleep_quality'] * data['hrv_mean'] / 50
    
    return data

data = generate_health_data(n_samples)
print(f"Generated {len(data)} samples")
print(f"Risk distribution: Low={sum(data['risk_level']==0)}, High={sum(data['risk_level']==1)}")

# Step 2: Prepare features
print("\n2. Preparing features...")
feature_cols = [
    'average_heart_rate', 'hrv_mean', 'respiratory_rate', 
    'activity_level', 'sleep_quality', 'stress_indicator',
    'hr_hrv_ratio', 'recovery_score'
]

X = data[feature_cols]
y = data['risk_level']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Training set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")

# Step 3: Create and train model
print("\n3. Creating gradient boosting model...")

# Use sklearn's GradientBoostingClassifier instead of XGBoost
gbm_model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=4,
    random_state=42,
    subsample=0.8
)

# Pipeline with scaling
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('gbm', gbm_model)
])

# Train the model
print("\n4. Training the model...")
pipeline.fit(X_train, y_train)
print("Training completed!")

# Step 4: Evaluate performance
print("\n5. Evaluating model performance...")

y_pred = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:, 1]

accuracy = pipeline.score(X_test, y_test)
auc_score = roc_auc_score(y_test, y_proba)

print(f"\nTest Set Performance:")
print(f"Accuracy: {accuracy:.3f}")
print(f"AUC Score: {auc_score:.3f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, 
                          target_names=['Low Risk', 'High Risk'],
                          digits=3))

# Feature importance
print("\n6. Feature importance:")
feature_importance = gbm_model.feature_importances_
for feat, imp in sorted(zip(feature_cols, feature_importance), 
                       key=lambda x: x[1], reverse=True):
    print(f"{feat}: {imp:.3f}")

# Step 5: Save model and metadata
print("\n7. Saving model and metadata...")

model_path = '/home/johaan/Documents/GitHub/TelemetryHealthCare/gbm_health_risk_model.pkl'
joblib.dump(pipeline, model_path)
print(f"Model saved to: {model_path}")

metadata = {
    'model_type': 'Gradient Boosting Classifier',
    'purpose': 'Health risk assessment without blood pressure',
    'input_features': {
        'average_heart_rate': 'From Apple Watch continuous monitoring',
        'hrv_mean': 'Heart rate variability from HealthKit',
        'respiratory_rate': 'From Apple Watch sleep monitoring',
        'activity_level': 'Steps/activity from Apple Watch',
        'sleep_quality': 'Sleep analysis from Apple Watch',
        'stress_indicator': 'Derived from HRV patterns',
        'hr_hrv_ratio': 'Calculated ratio',
        'recovery_score': 'Sleep and HRV based recovery metric'
    },
    'apple_watch_compatible': True,
    'blood_pressure_required': False,
    'performance': {
        'accuracy': float(accuracy),
        'auc_score': float(auc_score)
    },
    'training_date': datetime.now().isoformat(),
    'training_samples': len(X_train)
}

metadata_path = '/home/johaan/Documents/GitHub/TelemetryHealthCare/gbm_model_metadata.json'
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)
print(f"Metadata saved to: {metadata_path}")

# Step 6: Test with sample data
print("\n8. Testing with sample Apple Watch data...")
sample_data = pd.DataFrame({
    'average_heart_rate': [72, 95, 58],
    'hrv_mean': [55, 25, 40],
    'respiratory_rate': [14, 20, 12],
    'activity_level': [300, 50, 200],
    'sleep_quality': [0.8, 0.4, 0.7],
    'stress_indicator': [0.3, 0.8, 0.5]
})

# Add derived features
sample_data['hr_hrv_ratio'] = sample_data['average_heart_rate'] / (sample_data['hrv_mean'] + 1)
sample_data['recovery_score'] = sample_data['sleep_quality'] * sample_data['hrv_mean'] / 50

predictions = pipeline.predict(sample_data)
probabilities = pipeline.predict_proba(sample_data)[:, 1]

print("\nSample predictions:")
for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
    risk = "High Risk" if pred == 1 else "Low Risk"
    print(f"Sample {i+1}: {risk} (confidence: {prob:.2%})")

print("\n✅ GBM model training completed successfully!")
print("Ready for Apple Watch health monitoring without blood pressure data")