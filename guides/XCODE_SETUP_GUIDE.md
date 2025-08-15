# Xcode Setup Guide - Making Everything Work

## 🚨 Why Some Files Don't Show in Xcode

Xcode only shows iOS-related files by default:
- ✅ Swift files (.swift)
- ✅ Storyboards, XIBs
- ✅ Assets
- ❌ Python files (.py)
- ❌ Notebooks (.ipynb)
- ❌ Model files (.pkl)

## 📱 Step 1: Convert Models for iOS

**IMPORTANT**: The `.pkl` files must be converted to `.mlmodel` format!

### On your Mac Terminal:
```bash
# 1. Install coremltools
pip install coremltools

# 2. Navigate to project folder
cd /path/to/TelemetryHealthCare

# 3. Run the conversion script
python3 convert_models_to_coreml.py
```

This creates 3 new files:
- `HeartRhythmClassifier.mlmodel`
- `HealthRiskAssessment.mlmodel`
- `HRVPatternAnalyzer.mlmodel`

## 📂 Step 2: Add Models to Xcode

1. **Drag the .mlmodel files into Xcode**
   - Find the 3 `.mlmodel` files in Finder
   - Drag them into the Xcode project navigator
   - ✅ Check "Copy items if needed"
   - ✅ Check your app target

2. **Xcode will automatically generate Swift classes!**
   - You'll see the model files with a brain icon 🧠
   - Click on each to see inputs/outputs

## 🔧 Step 3: Update Your Swift Code

Replace the mock `ModelManager.swift` with the new `ModelManager_CoreML.swift`:

1. Delete or rename the old ModelManager.swift
2. Add ModelManager_CoreML.swift to your project
3. Rename it to ModelManager.swift

## 📄 Step 4: Add Documentation to Xcode (Optional)

To see documentation in Xcode:
1. Right-click your project
2. "Add Files to TelemetryHealthCare"
3. Select:
   - NEXT_STEPS_GUIDE.md
   - PROJECT_FILE_GUIDE.md
   - README.md
4. **UNCHECK** "Copy items if needed"
5. Click Add

## ✅ Verification Checklist

After setup, you should see in Xcode:
- [ ] 3 .mlmodel files (with brain icons)
- [ ] All Swift files
- [ ] Documentation files (if added)
- [ ] No errors when building

## 🏃‍♂️ Quick Test

1. Build the project (⌘+B)
2. Run on simulator (⌘+R)
3. Grant HealthKit permissions
4. You should see "✓ All Core ML models loaded successfully" in console

## 🆘 Troubleshooting

**"Cannot find type 'HeartRhythmClassifier' in scope"**
- Make sure .mlmodel files are added to your target
- Clean build folder (⌘+Shift+K) and rebuild

**"Python command not found"**
- Install Python 3: `brew install python3`
- Install pip: `python3 -m ensurepip`

**Models not converting**
- Check you're in the right directory
- Ensure .pkl files are present
- Try: `pip install --upgrade coremltools scikit-learn`

## 📱 Final Project Structure in Xcode

```
TelemetryHealthCare/
├── Models/
│   ├── HeartRhythmClassifier.mlmodel
│   ├── HealthRiskAssessment.mlmodel
│   └── HRVPatternAnalyzer.mlmodel
├── Views/
│   └── ContentView.swift
├── Managers/
│   ├── HealthKitManager.swift
│   └── ModelManager.swift
└── TelemetryHealthCareApp.swift
```

Now your Xcode project has everything it needs! 🎉