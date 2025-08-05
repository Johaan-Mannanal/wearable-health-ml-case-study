#!/usr/bin/env python3
"""
Convert trained scikit-learn models to Core ML format for iOS
Run this script to create .mlmodel files that Xcode can use
"""

import joblib
import coremltools as ct
import numpy as np

print("Converting ML models to Core ML format for iOS...")
print("=" * 60)

# Check if coremltools is installed
try:
    import coremltools
    print("✓ CoreML Tools is installed")
except ImportError:
    print("❌ Please install coremltools first:")
    print("   pip install coremltools")
    exit(1)

# Convert SVM Model
print("\n1. Converting SVM Heart Rhythm Model...")
try:
    # Load the model
    svm_model = joblib.load('svm_heart_rhythm_model.pkl')
    
    # Convert to Core ML
    svm_coreml = ct.converters.sklearn.convert(
        svm_model,
        input_features=[
            ('mean_heart_rate', 'double'),
            ('std_heart_rate', 'double'),
            ('pnn50', 'double')
        ],
        output_feature_names=['rhythm_prediction', 'rhythm_probability']
    )
    
    # Add metadata
    svm_coreml.author = 'TelemetryHealthCare'
    svm_coreml.short_description = 'Detects irregular heart rhythms from Apple Watch data'
    svm_coreml.input_description['mean_heart_rate'] = 'Average heart rate from HealthKit'
    svm_coreml.input_description['std_heart_rate'] = 'Heart rate standard deviation'
    svm_coreml.input_description['pnn50'] = 'HRV metric (percentage of NN intervals > 50ms)'
    
    # Save the model
    svm_coreml.save('HeartRhythmClassifier.mlmodel')
    print("✓ Saved: HeartRhythmClassifier.mlmodel")
    
except Exception as e:
    print(f"❌ Error converting SVM model: {e}")

# Convert GBM Model
print("\n2. Converting GBM Health Risk Model...")
try:
    # Load the model
    gbm_model = joblib.load('gbm_health_risk_model.pkl')
    
    # Convert to Core ML
    gbm_coreml = ct.converters.sklearn.convert(
        gbm_model,
        input_features=[
            ('average_heart_rate', 'double'),
            ('hrv_mean', 'double'),
            ('respiratory_rate', 'double'),
            ('activity_level', 'double'),
            ('sleep_quality', 'double'),
            ('stress_indicator', 'double'),
            ('hr_hrv_ratio', 'double'),
            ('recovery_score', 'double')
        ],
        output_feature_names=['risk_prediction', 'risk_probability']
    )
    
    # Add metadata
    gbm_coreml.author = 'TelemetryHealthCare'
    gbm_coreml.short_description = 'Assesses health risk without blood pressure'
    
    # Save the model
    gbm_coreml.save('HealthRiskAssessment.mlmodel')
    print("✓ Saved: HealthRiskAssessment.mlmodel")
    
except Exception as e:
    print(f"❌ Error converting GBM model: {e}")

# Convert Neural Network Model
print("\n3. Converting Neural Network HRV Model...")
try:
    # Load the model
    nn_model = joblib.load('hrv_pattern_nn_model.pkl')
    
    # Convert to Core ML
    nn_coreml = ct.converters.sklearn.convert(
        nn_model,
        input_features=[
            ('mean_rr', 'double'),
            ('std_rr', 'double'),
            ('min_rr', 'double'),
            ('max_rr', 'double'),
            ('q25_rr', 'double'),
            ('q75_rr', 'double'),
            ('mean_diff_rr', 'double'),
            ('std_diff_rr', 'double'),
            ('rmssd', 'double'),
            ('pnn50', 'double'),
            ('low_freq_power', 'double'),
            ('mid_freq_power', 'double'),
            ('high_freq_power', 'double')
        ],
        output_feature_names=['pattern_prediction', 'pattern_probabilities']
    )
    
    # Add metadata
    nn_coreml.author = 'TelemetryHealthCare'
    nn_coreml.short_description = 'Classifies HRV patterns (Normal, AFib, Bradycardia, Tachycardia)'
    
    # Save the model
    nn_coreml.save('HRVPatternAnalyzer.mlmodel')
    print("✓ Saved: HRVPatternAnalyzer.mlmodel")
    
except Exception as e:
    print(f"❌ Error converting Neural Network model: {e}")

print("\n" + "=" * 60)
print("✅ Conversion complete!")
print("\nNext steps:")
print("1. The .mlmodel files are ready for Xcode")
print("2. Drag the .mlmodel files into your Xcode project")
print("3. Make sure 'Copy items if needed' is checked")
print("4. Xcode will automatically generate Swift classes for each model")
print("\nYour models will appear in Xcode as:")
print("  - HeartRhythmClassifier")
print("  - HealthRiskAssessment") 
print("  - HRVPatternAnalyzer")