#!/usr/bin/env python3
"""
Test script for the improved SVM model.

This script validates the improvements made to the heart rhythm classification model
and compares performance against the original implementation.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

def generate_original_data(num_samples=5000):
    """
    Generate data using the original method for comparison.
    """
    np.random.seed(42)
    
    mean_heart_rate = np.random.normal(loc=75, scale=5, size=num_samples)
    std_heart_rate = np.random.normal(loc=5, scale=2, size=num_samples)
    pnn50 = np.random.uniform(0, 0.3, size=num_samples)
    target = np.random.choice([0, 1], size=num_samples, p=[0.6, 0.4])
    
    return pd.DataFrame({
        'mean_heart_rate': mean_heart_rate,
        'std_heart_rate': std_heart_rate,
        'pnn50': pnn50,
        'target': target
    })

def generate_improved_data(num_samples=5000):
    """
    Generate data using the improved method with realistic feature relationships.
    """
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

def train_original_model(data):
    """
    Train model using original approach.
    """
    X = data.drop('target', axis=1)
    y = data['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Original model setup
    model = SVC(kernel='rbf', probability=True, C=10, gamma='auto', class_weight={0:1, 1:2})
    model.fit(X_train, y_train)
    
    # Make predictions with original threshold adjustment
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba > 0.3).astype(int)
    
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    return {
        'model': model,
        'accuracy': accuracy,
        'auc': auc,
        'y_test': y_test,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }

def train_improved_model(data):
    """
    Train model using improved approach.
    """
    X = data.drop('target', axis=1)
    y = data['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Improved preprocessing
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Individual models
    svm_model = SVC(kernel='rbf', probability=True, C=10, gamma='scale', class_weight='balanced', random_state=42)
    lr_model = LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000)
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    
    # Ensemble model
    ensemble_model = VotingClassifier(
        estimators=[
            ('svm', svm_model),
            ('lr', lr_model),
            ('rf', rf_model)
        ],
        voting='soft'
    )
    
    # Train models
    svm_model.fit(X_train_scaled, y_train)
    lr_model.fit(X_train_scaled, y_train)
    rf_model.fit(X_train_scaled, y_train)
    ensemble_model.fit(X_train_scaled, y_train)
    
    # Evaluate ensemble model
    y_pred = ensemble_model.predict(X_test_scaled)
    y_pred_proba = ensemble_model.predict_proba(X_test_scaled)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    return {
        'model': ensemble_model,
        'scaler': scaler,
        'accuracy': accuracy,
        'auc': auc,
        'y_test': y_test,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }

def main():
    """
    Run comprehensive comparison between original and improved models.
    """
    print("🧪 Testing Improved Heart Rhythm Classification Model")
    print("=" * 60)
    
    # Test 1: Original data with original model
    print("\n1️⃣  Testing Original Model with Original Data Generation")
    print("-" * 50)
    original_data = generate_original_data(5000)
    original_results = train_original_model(original_data)
    
    print(f"Original Model Performance:")
    print(f"  Accuracy: {original_results['accuracy']:.3f}")
    print(f"  AUC Score: {original_results['auc']:.3f}")
    
    # Test 2: Improved data with improved model
    print("\n2️⃣  Testing Improved Model with Improved Data Generation")
    print("-" * 50)
    improved_data = generate_improved_data(5000)
    improved_results = train_improved_model(improved_data)
    
    print(f"Improved Model Performance:")
    print(f"  Accuracy: {improved_results['accuracy']:.3f}")
    print(f"  AUC Score: {improved_results['auc']:.3f}")
    
    # Test 3: Cross-comparison
    print("\n3️⃣  Cross-Comparison Analysis")
    print("-" * 50)
    
    # Original model on improved data
    original_on_improved = train_original_model(improved_data)
    print(f"Original Model on Improved Data:")
    print(f"  Accuracy: {original_on_improved['accuracy']:.3f}")
    print(f"  AUC Score: {original_on_improved['auc']:.3f}")
    
    # Test 4: Performance Summary
    print("\n📊 Performance Summary")
    print("=" * 60)
    
    performance_data = {
        'Configuration': [
            'Original Model + Original Data',
            'Original Model + Improved Data', 
            'Improved Model + Improved Data'
        ],
        'Accuracy': [
            original_results['accuracy'],
            original_on_improved['accuracy'],
            improved_results['accuracy']
        ],
        'AUC Score': [
            original_results['auc'],
            original_on_improved['auc'],
            improved_results['auc']
        ]
    }
    
    df_performance = pd.DataFrame(performance_data)
    print(df_performance.to_string(index=False, float_format='%.3f'))
    
    # Calculate improvements
    accuracy_improvement = (improved_results['accuracy'] - original_results['accuracy']) / original_results['accuracy'] * 100
    auc_improvement = (improved_results['auc'] - original_results['auc']) / original_results['auc'] * 100
    
    print(f"\n🚀 Overall Improvements:")
    print(f"  Accuracy Improvement: {accuracy_improvement:+.1f}%")
    print(f"  AUC Score Improvement: {auc_improvement:+.1f}%")
    
    # Test 5: Feature Analysis
    print("\n📈 Feature Analysis (Improved Data)")
    print("-" * 50)
    
    print("Feature correlations with target:")
    for feature in ['mean_heart_rate', 'std_heart_rate', 'pnn50']:
        corr = improved_data[feature].corr(improved_data['target'])
        print(f"  {feature}: {corr:.3f}")
    
    print("\nFeature statistics by class:")
    print(improved_data.groupby('target')[['mean_heart_rate', 'std_heart_rate', 'pnn50']].mean().round(3))
    
    # Test 6: Clinical Relevance Check
    print("\n🏥 Clinical Relevance Assessment")
    print("-" * 50)
    
    # Check if the model achieves clinically acceptable performance
    if improved_results['auc'] >= 0.8:
        clinical_grade = "Excellent (AUC ≥ 0.8)"
    elif improved_results['auc'] >= 0.7:
        clinical_grade = "Good (AUC ≥ 0.7)"
    elif improved_results['auc'] >= 0.6:
        clinical_grade = "Fair (AUC ≥ 0.6)"
    else:
        clinical_grade = "Poor (AUC < 0.6)"
    
    print(f"Clinical Performance Grade: {clinical_grade}")
    
    # Sensitivity and Specificity Analysis
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(improved_results['y_test'], improved_results['y_pred'])
    
    if len(cm) == 2:
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        print(f"Sensitivity (Recall): {sensitivity:.3f}")
        print(f"Specificity: {specificity:.3f}")
        
        if sensitivity >= 0.8 and specificity >= 0.8:
            print("✅ Model meets clinical thresholds for both sensitivity and specificity")
        elif sensitivity >= 0.8:
            print("⚠️  Good sensitivity but specificity could be improved")
        elif specificity >= 0.8:
            print("⚠️  Good specificity but sensitivity could be improved")
        else:
            print("❌ Both sensitivity and specificity need improvement")
    
    print(f"\n✅ Model testing completed!")
    print(f"📁 Results ready for HealthKit integration")
    
    return {
        'original_results': original_results,
        'improved_results': improved_results,
        'performance_summary': df_performance
    }

if __name__ == "__main__":
    results = main()