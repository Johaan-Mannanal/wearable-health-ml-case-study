# Project File Guide - What Every File Does

> ⚠️ Synthetic-data research project — not a medical device. See the root README.md and MODEL_CARD.md.

## 📁 Complete File List with Explanations

### 🤖 Machine Learning Models
These models are trained and evaluated on **synthetic** data. Accuracies are synthetic-data figures
(verified in `results/metrics.csv`), and all labels are generated constructs, not medical findings:

1. **svm_heart_rhythm_model.pkl** (93.9% accuracy, synthetic)
   - **What it does**: classifies a synthetic Normal/Irregular rhythm label
   - **Inputs**: Average heart rate, heart rate variability, pNN50
   - **Note**: a classification demo, not arrhythmia detection

2. **gbm_health_risk_model.pkl** (99.4% accuracy, synthetic)
   - **What it does**: classifies a synthetic Low/High "risk" label
   - **Inputs**: Heart rate, HRV, activity, sleep, stress (no blood pressure input)
   - **Note**: separable-by-design synthetic classes

3. **hrv_pattern_nn_model.pkl** (99.0% accuracy, synthetic)
   - **What it does**: classifies 4 synthetic HRV pattern labels (normal, afib, bradycardia, tachycardia)
   - **Inputs**: 50 heart rate readings (~12 seconds of data)
   - **Note**: labels are synthetic classes, not diagnoses

### 📊 Model Metadata Files (Model Information)
These JSON files contain important information about each model:

4. **svm_model_metadata.json**
   - Contains: Model accuracy, feature names, HealthKit mappings for SVM

5. **gbm_model_metadata.json**
   - Contains: Model specs, feature list, performance metrics for GBM

6. **hrv_nn_model_metadata.json**
   - Contains: Neural network architecture, input requirements

7. **model_metadata.json**
   - Contains: General model information and specifications

### 📓 Jupyter Notebooks (Development History)

**Original Notebooks** (Keep for reference):
8. **Convolutional_Neural_Network.ipynb**
   - Original CNN for ECG data (not compatible with Apple Watch)

9. **Gradient_Boosting_Machine.ipynb**
   - Original GBM requiring blood pressure

10. **Support_Vector_Machine.ipynb**
    - Original SVM with 41% accuracy

**Improved Notebooks** (What we actually use):
11. **Support_Vector_Machine_Improved.ipynb**
    - Enhanced SVM with 93% accuracy, Apple Watch compatible

12. **Enhanced_Gradient_Boosting_Machine.ipynb**
    - GBM without blood pressure requirement

13. **HRV_CNN_Analysis.ipynb**
    - Redesigned CNN using HRV instead of ECG

### 🐍 Python Scripts (Training & Testing)

14. **train_svm_model.py**
    - Script to retrain the SVM model with new data

15. **train_gbm_model.py**
    - Script to retrain the GBM model

16. **train_hrv_nn_model.py**
    - Script to retrain the neural network

17. **test_all_models.py**
    - Tests all three models with synthetic data

18. **test_svm_detailed.py**
    - Detailed testing for SVM model

19. **test_improved_model.py**
    - Tests the improved models

20. **visualize_test_results.py**
    - Creates visualizations of test results

21. **healthkit_data_processor.py** ⭐
    - Helper for converting HealthKit-style data into model inputs (for the WIP app)
    - Maps HealthKit fields to model features
    - Generates illustrative (non-diagnostic) summary output

22. **create_model_pipeline.py**
    - Creates the complete ML pipeline

### 📱 iOS App Files (The App Interface)

**Main App Code**:
23. **TelemetryHealthCareApp.swift**
    - The main app entry point

24. **ContentView.swift**
    - The main screen you see (shows heart rate, risk level)

25. **HealthKitManager.swift**
    - Connects to your Apple Watch and reads health data

26. **Info.plist**
    - App configuration and permissions

27. **TelemetryHealthCare.entitlements**
    - Security settings for HealthKit access

**App Resources**:
28. **Assets.xcassets/** (folder)
    - App icons and colors
    - Contains: AppIcon, AccentColor settings

**Test Files**:
29. **TelemetryHealthCareTests.swift**
    - Unit tests for the app

30. **TelemetryHealthCareUITests.swift**
    - UI tests for the app

31. **TelemetryHealthCareUITestsLaunchTests.swift**
    - Launch tests for the app

### 📚 Documentation (Start Here!)

32. **NEXT_STEPS_GUIDE.md** ⭐⭐⭐
    - **START HERE**: Simple instructions for completing the project
    - Step-by-step guide for non-technical users

33. **iOS_CODE_TEMPLATES.md** ⭐⭐
    - Copy-paste Swift code to get started quickly
    - Includes Core ML conversion script

34. **README.md**
    - Basic project introduction

35. **TEST_REPORT.md**
    - Detailed test results showing models work correctly

36. **TRAINED_MODELS_SUMMARY.md**
    - Summary of all trained models and performance

37. **MODEL_IMPROVEMENTS_SUMMARY.md**
    - Technical details of model improvements

38. **Enhanced_Model_Documentation.md**
    - Documentation for the enhanced GBM model

39. **Research Project Outline.txt**
    - Original project research outline

### 🔧 Configuration Files

40. **requirements.txt**
    - Python packages needed (numpy, pandas, scikit-learn, etc.)
    - Run: `pip install -r requirements.txt`

41. **TelemetryHealthCare.xcodeproj/** (folder)
    - Xcode project configuration
    - Contains: project.pbxproj, workspace settings

## 🚀 Quick Start Guide

### For Developers:
1. Read `NEXT_STEPS_GUIDE.md` first
2. Look at `iOS_CODE_TEMPLATES.md` for Swift code
3. Use `healthkit_data_processor.py` for data processing
4. Models are in `.pkl` files

### For Users:
1. Follow `NEXT_STEPS_GUIDE.md` step by step
2. The app uses the 3 `.pkl` model files automatically
3. Your health data flows: Apple Watch → HealthKit → App → Models → Results

### For Researchers:
1. Notebooks show the complete ML development process
2. Test scripts demonstrate model performance
3. Training scripts allow model updates

## ❓ Common Questions

**Q: Which files do I actually need to run the app?**
A: The `.pkl` models, iOS Swift files, and the Xcode project

**Q: Can I delete the notebooks?**
A: Keep them for reference, but they're not needed to run the app

**Q: What's the most important file?**
A: `NEXT_STEPS_GUIDE.md` - it explains everything in simple terms

**Q: Where's the code that connects to my Apple Watch?**
A: `HealthKitManager.swift` and `healthkit_data_processor.py`