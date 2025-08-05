#!/usr/bin/env python3
"""
Train the improved SVM model for heart rhythm classification
Compatible with Apple Watch Series 10 data via HealthKit
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.pipeline import Pipeline
import joblib
import json
from datetime import datetime

print("Starting SVM model training for Apple Watch heart rhythm classification...")

# Step 1: Generate synthetic training data (5000 samples)
print("\n1. Generating synthetic training data...")
np.random.seed(42)
n_samples = 5000

# Create realistic heart rhythm data
def generate_realistic_data(n_samples):
    # Normal rhythm patterns (60% of data)
    n_normal = int(n_samples * 0.6)
    normal_data = pd.DataFrame({
        'mean_heart_rate': np.random.normal(70, 10, n_normal),
        'std_heart_rate': np.random.gamma(2, 1.5, n_normal),
        'pnn50': np.random.beta(2, 5, n_normal) * 0.5
    })
    normal_data['label'] = 0
    
    # Irregular rhythm patterns (40% of data)
    n_irregular = n_samples - n_normal
    irregular_data = pd.DataFrame({
        'mean_heart_rate': np.concatenate([
            np.random.normal(95, 15, n_irregular//2),  # Tachycardia
            np.random.normal(50, 8, n_irregular//2)    # Bradycardia
        ]),
        'std_heart_rate': np.random.gamma(4, 2, n_irregular),
        'pnn50': np.random.beta(1, 8, n_irregular) * 0.3
    })
    irregular_data['label'] = 1
    
    # Combine and add realistic correlations
    data = pd.concat([normal_data, irregular_data], ignore_index=True)
    
    # Add physiological constraints
    data['mean_heart_rate'] = np.clip(data['mean_heart_rate'], 40, 120)
    data['std_heart_rate'] = np.clip(data['std_heart_rate'], 1, 20)
    data['pnn50'] = np.clip(data['pnn50'], 0, 0.5)
    
    return data

data = generate_realistic_data(n_samples)
print(f"Generated {len(data)} samples")
print(f"Class distribution: Normal={sum(data['label']==0)}, Irregular={sum(data['label']==1)}")

# Step 2: Prepare features and split data
print("\n2. Preparing features and splitting data...")
X = data[['mean_heart_rate', 'std_heart_rate', 'pnn50']]
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Training set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")

# Step 3: Create ensemble model pipeline
print("\n3. Creating ensemble model pipeline...")

# Individual models
svm_model = SVC(kernel='rbf', C=10, gamma=0.1, probability=True, random_state=42)
lr_model = LogisticRegression(max_iter=1000, random_state=42)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Ensemble with soft voting
ensemble = VotingClassifier(
    estimators=[
        ('svm', svm_model),
        ('lr', lr_model),
        ('rf', rf_model)
    ],
    voting='soft'
)

# Complete pipeline with scaling
pipeline = Pipeline([
    ('scaler', RobustScaler()),
    ('classifier', ensemble)
])

# Step 4: Train the model
print("\n4. Training the ensemble model...")
pipeline.fit(X_train, y_train)
print("Training completed!")

# Step 5: Evaluate performance
print("\n5. Evaluating model performance...")

# Test set predictions
y_pred = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:, 1]

# Metrics
accuracy = pipeline.score(X_test, y_test)
auc_score = roc_auc_score(y_test, y_proba)

print(f"\nTest Set Performance:")
print(f"Accuracy: {accuracy:.3f}")
print(f"AUC Score: {auc_score:.3f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, 
                          target_names=['Normal', 'Irregular'],
                          digits=3))

# Cross-validation
print("\n6. Cross-validation (5-fold)...")
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy')
print(f"CV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

# Step 6: Save the model and metadata
print("\n7. Saving model and metadata...")

# Save the trained model
model_path = '/home/johaan/Documents/GitHub/TelemetryHealthCare/svm_heart_rhythm_model.pkl'
joblib.dump(pipeline, model_path)
print(f"Model saved to: {model_path}")

# Save model metadata
metadata = {
    'model_type': 'SVM Ensemble (SVM + Logistic Regression + Random Forest)',
    'purpose': 'Heart rhythm classification (Normal vs Irregular)',
    'input_features': {
        'mean_heart_rate': 'Average heart rate from HealthKit',
        'std_heart_rate': 'Heart rate standard deviation',
        'pnn50': 'Percentage of NN intervals > 50ms'
    },
    'apple_watch_compatible': True,
    'performance': {
        'accuracy': float(accuracy),
        'auc_score': float(auc_score),
        'cv_accuracy': float(cv_scores.mean())
    },
    'training_date': datetime.now().isoformat(),
    'training_samples': len(X_train),
    'healthkit_mapping': {
        'mean_heart_rate': 'HKQuantityTypeIdentifierHeartRate',
        'std_heart_rate': 'HKQuantityTypeIdentifierHeartRateVariabilitySDNN',
        'pnn50': 'Derived from HKQuantityTypeIdentifierHeartRateVariabilityRMSSD'
    }
}

metadata_path = '/home/johaan/Documents/GitHub/TelemetryHealthCare/svm_model_metadata.json'
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)
print(f"Metadata saved to: {metadata_path}")

# Step 7: Test with sample Apple Watch-like data
print("\n8. Testing with sample Apple Watch-like data...")
sample_data = pd.DataFrame({
    'mean_heart_rate': [72, 95, 55],
    'std_heart_rate': [5.2, 12.5, 3.8],
    'pnn50': [0.15, 0.05, 0.25]
})

predictions = pipeline.predict(sample_data)
probabilities = pipeline.predict_proba(sample_data)[:, 1]

print("\nSample predictions:")
for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
    rhythm = "Irregular" if pred == 1 else "Normal"
    print(f"Sample {i+1}: {rhythm} (confidence: {prob:.2%})")
    
print("\n✅ SVM model training completed successfully!")
print("Ready for integration with Apple Watch data via HealthKit")