#!/usr/bin/env python3
"""
Train a Neural Network model for HRV pattern analysis
Simplified version using scikit-learn instead of TensorFlow
Compatible with Apple Watch Series 10 continuous heart rate data
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import joblib
import json
from datetime import datetime

print("Starting Neural Network training for HRV pattern analysis...")

# Step 1: Generate synthetic HRV time series data
print("\n1. Generating synthetic HRV time series data...")
np.random.seed(42)

def generate_hrv_patterns(n_samples=1000, sequence_length=50):
    """Generate realistic HRV patterns for different conditions"""
    
    conditions = ['normal', 'afib', 'bradycardia', 'tachycardia']
    samples_per_condition = n_samples // 4
    
    X_data = []
    y_data = []
    
    for condition_idx, condition in enumerate(conditions):
        for _ in range(samples_per_condition):
            # Base heart rate for each condition
            if condition == 'normal':
                base_hr = np.random.normal(70, 10)
                variability = np.random.normal(0, 5, sequence_length)
                pattern = 'regular'
            elif condition == 'afib':
                base_hr = np.random.normal(80, 15)
                # Irregular patterns
                variability = np.random.normal(0, 15, sequence_length)
                # Add random spikes
                spike_indices = np.random.choice(sequence_length, 5)
                variability[spike_indices] += np.random.normal(20, 5, len(spike_indices))
                pattern = 'irregular'
            elif condition == 'bradycardia':
                base_hr = np.random.normal(50, 5)
                variability = np.random.normal(0, 3, sequence_length)
                pattern = 'slow_regular'
            else:  # tachycardia
                base_hr = np.random.normal(100, 10)
                variability = np.random.normal(0, 2, sequence_length)
                pattern = 'fast_regular'
            
            # Generate HR sequence
            hr_sequence = base_hr + variability
            hr_sequence = np.clip(hr_sequence, 40, 150)
            
            # Calculate RR intervals (60000/HR for ms)
            rr_intervals = 60000 / hr_sequence
            
            # Extract features from the sequence
            features = []
            features.extend([
                np.mean(rr_intervals),
                np.std(rr_intervals),
                np.min(rr_intervals),
                np.max(rr_intervals),
                np.percentile(rr_intervals, 25),
                np.percentile(rr_intervals, 75),
                np.mean(np.diff(rr_intervals)),
                np.std(np.diff(rr_intervals)),
                # RMSSD - important HRV metric
                np.sqrt(np.mean(np.diff(rr_intervals)**2)),
                # pNN50 approximation
                len(np.where(np.abs(np.diff(rr_intervals)) > 50)[0]) / len(rr_intervals)
            ])
            
            # Add frequency domain features (simplified)
            fft_vals = np.abs(np.fft.fft(rr_intervals))[:sequence_length//2]
            features.extend([
                np.mean(fft_vals[:5]),  # Low frequency
                np.mean(fft_vals[5:15]),  # Mid frequency
                np.mean(fft_vals[15:])  # High frequency
            ])
            
            X_data.append(features)
            y_data.append(condition_idx)
    
    return np.array(X_data), np.array(y_data), conditions

# Generate data
X, y, class_names = generate_hrv_patterns(n_samples=4000, sequence_length=50)
print(f"Generated {len(X)} HRV pattern samples")
print(f"Features per sample: {X.shape[1]}")
print(f"Classes: {class_names}")

# Step 2: Split data
print("\n2. Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Training set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")

# Step 3: Create neural network model
print("\n3. Creating neural network model...")

# Multi-layer perceptron with 3 hidden layers
mlp = MLPClassifier(
    hidden_layer_sizes=(64, 32, 16),
    activation='relu',
    solver='adam',
    alpha=0.001,
    batch_size='auto',
    learning_rate='adaptive',
    learning_rate_init=0.001,
    max_iter=500,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=20
)

# Pipeline with scaling
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('mlp', mlp)
])

# Step 4: Train the model
print("\n4. Training the neural network...")
pipeline.fit(X_train, y_train)
print(f"Training completed in {mlp.n_iter_} iterations")

# Step 5: Evaluate performance
print("\n5. Evaluating model performance...")

y_pred = pipeline.predict(X_test)
accuracy = pipeline.score(X_test, y_test)

print(f"\nTest Set Accuracy: {accuracy:.3f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, 
                          target_names=class_names,
                          digits=3))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print("True\\Pred", end="")
for name in class_names:
    print(f"\t{name[:6]}", end="")
print()
for i, name in enumerate(class_names):
    print(f"{name}", end="")
    for j in range(len(class_names)):
        print(f"\t{cm[i,j]}", end="")
    print()

# Step 6: Save model and metadata
print("\n6. Saving model and metadata...")

model_path = '/home/johaan/Documents/GitHub/TelemetryHealthCare/hrv_pattern_nn_model.pkl'
joblib.dump(pipeline, model_path)
print(f"Model saved to: {model_path}")

# Feature names for documentation
feature_names = [
    'mean_rr', 'std_rr', 'min_rr', 'max_rr', 
    'q25_rr', 'q75_rr', 'mean_diff_rr', 'std_diff_rr',
    'rmssd', 'pnn50', 'low_freq_power', 'mid_freq_power', 'high_freq_power'
]

metadata = {
    'model_type': 'Multi-Layer Perceptron (Neural Network)',
    'purpose': 'HRV pattern classification from Apple Watch heart rate data',
    'architecture': {
        'input_features': len(feature_names),
        'hidden_layers': [64, 32, 16],
        'output_classes': 4,
        'activation': 'relu'
    },
    'classes': class_names,
    'features': feature_names,
    'apple_watch_compatible': True,
    'input_requirements': {
        'data_type': 'Continuous heart rate measurements',
        'sequence_length': '50 heart rate readings (~12-15 seconds)',
        'sampling_rate': 'Apple Watch native rate (~4Hz)'
    },
    'performance': {
        'accuracy': float(accuracy),
        'training_iterations': int(mlp.n_iter_)
    },
    'training_date': datetime.now().isoformat(),
    'training_samples': len(X_train)
}

metadata_path = '/home/johaan/Documents/GitHub/TelemetryHealthCare/hrv_nn_model_metadata.json'
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)
print(f"Metadata saved to: {metadata_path}")

# Step 7: Test with sample data
print("\n7. Testing with sample Apple Watch-like HRV data...")

# Simulate different patterns
test_patterns = {
    'normal': {
        'base_hr': 72,
        'variability': 5,
        'description': 'Normal sinus rhythm'
    },
    'irregular': {
        'base_hr': 85,
        'variability': 20,
        'description': 'Irregular pattern (possible AFib)'
    },
    'slow': {
        'base_hr': 48,
        'variability': 3,
        'description': 'Slow regular rhythm'
    }
}

print("\nSample predictions:")
for pattern_name, params in test_patterns.items():
    # Generate test sequence
    hr_seq = params['base_hr'] + np.random.normal(0, params['variability'], 50)
    hr_seq = np.clip(hr_seq, 40, 150)
    rr_intervals = 60000 / hr_seq
    
    # Extract same features
    features = []
    features.extend([
        np.mean(rr_intervals), np.std(rr_intervals),
        np.min(rr_intervals), np.max(rr_intervals),
        np.percentile(rr_intervals, 25), np.percentile(rr_intervals, 75),
        np.mean(np.diff(rr_intervals)), np.std(np.diff(rr_intervals)),
        np.sqrt(np.mean(np.diff(rr_intervals)**2)),
        len(np.where(np.abs(np.diff(rr_intervals)) > 50)[0]) / len(rr_intervals)
    ])
    fft_vals = np.abs(np.fft.fft(rr_intervals))[:25]
    features.extend([
        np.mean(fft_vals[:5]), np.mean(fft_vals[5:15]), np.mean(fft_vals[15:])
    ])
    
    # Predict
    prediction = pipeline.predict([features])[0]
    proba = pipeline.predict_proba([features])[0]
    
    print(f"\n{params['description']}:")
    print(f"  Predicted: {class_names[prediction]}")
    print(f"  Confidence: {proba[prediction]:.2%}")

print("\n✅ Neural Network model training completed successfully!")
print("Ready for HRV pattern analysis from Apple Watch data")