#!/usr/bin/env python3
"""
Alternative Core ML conversion script with better error handling
Works around common sklearn/coremltools compatibility issues
"""

import sys
import os
import numpy as np

print("Alternative Core ML Model Conversion")
print("=" * 60)

# Check imports
try:
    import joblib
    print("✓ joblib imported")
except ImportError:
    print("❌ Please install joblib: pip3 install joblib")
    sys.exit(1)

try:
    import coremltools as ct
    print("✓ coremltools imported")
except ImportError:
    print("❌ Please install coremltools: pip3 install coremltools")
    sys.exit(1)

try:
    from sklearn import __version__ as sklearn_version
    print(f"✓ scikit-learn version: {sklearn_version}")
except ImportError:
    print("❌ Please install scikit-learn: pip3 install scikit-learn")
    sys.exit(1)

# Attempt conversions with error handling
successful_conversions = 0

# 1. Try SVM Model
print("\n1. Converting SVM Model...")
try:
    # Check if file exists
    if not os.path.exists('svm_heart_rhythm_model.pkl'):
        print("   ❌ svm_heart_rhythm_model.pkl not found!")
    else:
        # Try conversion
        svm_model = joblib.load('svm_heart_rhythm_model.pkl')
        print("   ✓ Model loaded successfully")
        
        # Alternative: Create a simple wrapper if direct conversion fails
        try:
            # Try direct conversion first
            svm_coreml = ct.converters.sklearn.convert(
                svm_model,
                input_features=['mean_heart_rate', 'std_heart_rate', 'pnn50'],
                output_feature_names='rhythm_class'
            )
            svm_coreml.save('HeartRhythmClassifier.mlmodel')
            print("   ✓ Successfully converted to HeartRhythmClassifier.mlmodel")
            successful_conversions += 1
        except Exception as e:
            print(f"   ⚠️  Direct conversion failed: {str(e)}")
            print("   💡 Try the manual conversion approach below")
            
except Exception as e:
    print(f"   ❌ Error: {str(e)}")

# 2. Try GBM Model
print("\n2. Converting GBM Model...")
try:
    if not os.path.exists('gbm_health_risk_model.pkl'):
        print("   ❌ gbm_health_risk_model.pkl not found!")
    else:
        gbm_model = joblib.load('gbm_health_risk_model.pkl')
        print("   ✓ Model loaded successfully")
        
        try:
            gbm_coreml = ct.converters.sklearn.convert(
                gbm_model,
                input_features=[
                    'average_heart_rate', 'hrv_mean', 'respiratory_rate',
                    'activity_level', 'sleep_quality', 'stress_indicator',
                    'hr_hrv_ratio', 'recovery_score'
                ],
                output_feature_names='risk_class'
            )
            gbm_coreml.save('HealthRiskAssessment.mlmodel')
            print("   ✓ Successfully converted to HealthRiskAssessment.mlmodel")
            successful_conversions += 1
        except Exception as e:
            print(f"   ⚠️  Direct conversion failed: {str(e)}")
            
except Exception as e:
    print(f"   ❌ Error: {str(e)}")

# 3. Try Neural Network Model
print("\n3. Converting Neural Network Model...")
try:
    if not os.path.exists('hrv_pattern_nn_model.pkl'):
        print("   ❌ hrv_pattern_nn_model.pkl not found!")
    else:
        nn_model = joblib.load('hrv_pattern_nn_model.pkl')
        print("   ✓ Model loaded successfully")
        
        try:
            # Neural network might be MLPClassifier which has better support
            nn_coreml = ct.converters.sklearn.convert(
                nn_model,
                input_features=[f'feature_{i}' for i in range(13)],
                output_feature_names='pattern_class'
            )
            nn_coreml.save('HRVPatternAnalyzer.mlmodel')
            print("   ✓ Successfully converted to HRVPatternAnalyzer.mlmodel")
            successful_conversions += 1
        except Exception as e:
            print(f"   ⚠️  Direct conversion failed: {str(e)}")
            
except Exception as e:
    print(f"   ❌ Error: {str(e)}")

# Summary
print("\n" + "=" * 60)
print(f"Conversion Summary: {successful_conversions}/3 models converted successfully")

if successful_conversions == 0:
    print("\n⚠️  No models were converted successfully.")
    print("\nTROUBLESHOOTING STEPS:")
    print("1. Check scikit-learn version compatibility:")
    print("   pip3 install scikit-learn==1.2.2")
    print("\n2. Try older coremltools:")
    print("   pip3 install coremltools==6.3")
    print("\n3. Manual Conversion Alternative:")
    print("   Since the models are relatively simple, you can:")
    print("   - Create the ML logic directly in Swift")
    print("   - Use the model parameters from the metadata JSON files")
    print("   - This avoids the conversion issue entirely")
    
    # Create a simple Swift implementation guide
    with open('MANUAL_MODEL_IMPLEMENTATION.md', 'w') as f:
        f.write("""# Manual Model Implementation in Swift

Since Core ML conversion is failing, here's how to implement the models directly in Swift:

## Option 1: Use Model Parameters Directly

The trained models are saved with their parameters. You can extract these and implement the logic in Swift.

## Option 2: Use a Server-Based Approach

1. Create a simple Python Flask/FastAPI server
2. Load the .pkl models on the server
3. Make API calls from the iOS app
4. This avoids Core ML entirely

## Option 3: Simplified Models

For the SVM (92% accuracy), you can approximate with simpler logic:
- If heart rate std > 15: Irregular
- If pNN50 < 0.1 and HR > 90: Irregular
- Otherwise: Normal

This won't be as accurate but works as a temporary solution.
""")
    print("\n📄 Created MANUAL_MODEL_IMPLEMENTATION.md with alternatives")

else:
    print(f"\n✅ Successfully converted {successful_conversions} models!")
    print("Next: Drag the .mlmodel files into Xcode")

# Additional diagnostics
print("\n📊 Diagnostic Information:")
print(f"Python version: {sys.version}")
print(f"Operating System: {os.uname().sysname}")
print(f"Current directory: {os.getcwd()}")
print(f"Files in directory: {len([f for f in os.listdir('.') if f.endswith('.pkl')])} .pkl files found")