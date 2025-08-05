#!/usr/bin/env python3
"""
Create and save the improved model pipeline for HealthKit integration.
This script generates the trained model that can be used by the HealthKit processor.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

def generate_improved_data(num_samples=10000):
    """Generate improved synthetic data for training."""
    np.random.seed(42)
    
    target = np.random.choice([0, 1], size=num_samples, p=[0.7, 0.3])
    
    mean_heart_rate = np.zeros(num_samples)
    std_heart_rate = np.zeros(num_samples)
    pnn50 = np.zeros(num_samples)
    
    for i in range(num_samples):
        if target[i] == 0:  # Normal rhythm
            mean_heart_rate[i] = np.random.normal(75, 8)
            std_heart_rate[i] = np.random.gamma(2, 2)
            pnn50[i] = np.random.beta(2, 8) * 0.4
        else:  # Irregular rhythm
            mean_heart_rate[i] = np.random.normal(85, 15)
            std_heart_rate[i] = np.random.gamma(3, 4)
            pnn50[i] = np.random.beta(3, 5) * 0.6
    
    # Add noise and ensure realistic bounds
    mean_heart_rate = np.clip(mean_heart_rate + np.random.normal(0, 2, num_samples), 40, 200)
    std_heart_rate = np.clip(std_heart_rate + np.random.normal(0, 1, num_samples), 0, 50)
    pnn50 = np.clip(pnn50 + np.random.normal(0, 0.02, num_samples), 0, 1)
    
    return pd.DataFrame({
        'mean_heart_rate': mean_heart_rate,
        'std_heart_rate': std_heart_rate,
        'pnn50': pnn50,
        'target': target
    })

def create_and_save_model():
    """Create, train, and save the improved model pipeline."""
    print("🔧 Creating Improved Heart Rhythm Classification Pipeline")
    print("=" * 60)
    
    # Generate training data
    print("📊 Generating training data...")
    data = generate_improved_data(10000)
    
    X = data.drop('target', axis=1)
    y = data['target']
    
    # Split data with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # Create preprocessing and model components
    print("🏗️  Building model components...")
    
    # Preprocessing
    scaler = RobustScaler()
    
    # Individual models
    svm_model = SVC(
        kernel='rbf', 
        probability=True, 
        C=10, 
        gamma='scale', 
        class_weight='balanced', 
        random_state=42
    )
    
    lr_model = LogisticRegression(
        random_state=42, 
        class_weight='balanced', 
        max_iter=1000
    )
    
    rf_model = RandomForestClassifier(
        n_estimators=100, 
        random_state=42, 
        class_weight='balanced'
    )
    
    # Create ensemble model
    ensemble_model = VotingClassifier(
        estimators=[
            ('svm', svm_model),
            ('lr', lr_model),
            ('rf', rf_model)
        ],
        voting='soft'
    )
    
    # Create complete pipeline
    pipeline = Pipeline([
        ('scaler', scaler),
        ('classifier', ensemble_model)
    ])
    
    # Train the pipeline
    print("🎯 Training model pipeline...")
    pipeline.fit(X_train, y_train)
    
    # Evaluate performance
    from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
    
    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"\n📈 Model Performance:")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"AUC Score: {auc:.3f}")
    
    # Save the complete pipeline
    pipeline_filename = 'improved_heart_rhythm_svm_pipeline.pkl'
    joblib.dump(pipeline, pipeline_filename)
    print(f"\n💾 Pipeline saved to: {pipeline_filename}")
    
    # Save individual components
    joblib.dump(ensemble_model, 'best_ensemble_model.pkl')
    joblib.dump(scaler, 'healthkit_data_scaler.pkl')
    
    # Create and save metadata
    # Get feature importance from the trained random forest in the ensemble
    try:
        # Access the trained random forest from the ensemble
        trained_rf = pipeline.named_steps['classifier'].named_estimators_['rf']
        feature_importance = trained_rf.feature_importances_
    except:
        # Fallback: train a separate RF to get feature importance
        temp_rf = RandomForestClassifier(n_estimators=100, random_state=42)
        X_train_scaled = scaler.fit_transform(X_train)
        temp_rf.fit(X_train_scaled, y_train)
        feature_importance = temp_rf.feature_importances_
    metadata = {
        'model_type': 'Ensemble',
        'auc_score': float(auc),
        'accuracy': float(accuracy),
        'features': ['mean_heart_rate', 'std_heart_rate', 'pnn50'],
        'feature_importance': feature_importance.tolist(),
        'training_samples': len(X_train),
        'test_samples': len(X_test),
        'healthkit_compatibility': {
            'mean_heart_rate': 'HKQuantityTypeIdentifierHeartRate',
            'std_heart_rate': 'HKQuantityTypeIdentifierHeartRateVariabilitySDNN',
            'pnn50': 'HKQuantityTypeIdentifierHeartRateVariabilityRMSSD'
        },
        'model_components': ['SVM', 'Logistic Regression', 'Random Forest'],
        'preprocessing': 'RobustScaler',
        'created_date': pd.Timestamp.now().isoformat()
    }
    
    with open('model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"📋 Metadata saved to: model_metadata.json")
    
    # Display detailed results
    print(f"\n📊 Detailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Irregular']))
    
    print(f"\n✅ Model pipeline creation completed!")
    print(f"📁 Files created:")
    print(f"   • {pipeline_filename}")
    print(f"   • best_ensemble_model.pkl")
    print(f"   • healthkit_data_scaler.pkl")
    print(f"   • model_metadata.json")
    
    return pipeline, metadata

if __name__ == "__main__":
    pipeline, metadata = create_and_save_model()