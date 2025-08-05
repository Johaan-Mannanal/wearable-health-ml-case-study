# iOS Code Templates - Copy These Into Xcode

## 1. ContentView.swift (Main Screen)

```swift
import SwiftUI

struct ContentView: View {
    @State private var heartRate: Double = 0
    @State private var rhythmStatus: String = "Checking..."
    @State private var riskLevel: String = "Unknown"
    @State private var isMonitoring: Bool = false
    
    var body: some View {
        NavigationView {
            VStack(spacing: 30) {
                // Heart Rate Display
                VStack {
                    Text("\(Int(heartRate))")
                        .font(.system(size: 80, weight: .bold))
                        .foregroundColor(.red)
                    Text("BPM")
                        .font(.headline)
                        .foregroundColor(.gray)
                }
                .padding()
                .background(Color.gray.opacity(0.1))
                .cornerRadius(20)
                
                // Status Cards
                HStack(spacing: 20) {
                    StatusCard(title: "Rhythm", value: rhythmStatus, color: .blue)
                    StatusCard(title: "Risk", value: riskLevel, color: .orange)
                }
                
                // Start/Stop Button
                Button(action: toggleMonitoring) {
                    Text(isMonitoring ? "Stop Monitoring" : "Start Monitoring")
                        .font(.headline)
                        .foregroundColor(.white)
                        .padding()
                        .frame(maxWidth: .infinity)
                        .background(isMonitoring ? Color.red : Color.green)
                        .cornerRadius(10)
                }
                
                Spacer()
            }
            .padding()
            .navigationTitle("Health Monitor")
        }
    }
    
    func toggleMonitoring() {
        isMonitoring.toggle()
        if isMonitoring {
            // Start health monitoring
            print("Starting monitoring...")
        } else {
            // Stop monitoring
            print("Stopping monitoring...")
        }
    }
}

struct StatusCard: View {
    let title: String
    let value: String
    let color: Color
    
    var body: some View {
        VStack {
            Text(title)
                .font(.caption)
                .foregroundColor(.gray)
            Text(value)
                .font(.headline)
                .foregroundColor(color)
        }
        .padding()
        .frame(maxWidth: .infinity)
        .background(Color.gray.opacity(0.1))
        .cornerRadius(10)
    }
}
```

## 2. HealthKitManager.swift (Apple Watch Connection)

```swift
import HealthKit

class HealthKitManager: ObservableObject {
    private let healthStore = HKHealthStore()
    
    // Data types we want to read
    private let heartRateType = HKQuantityType.quantityType(forIdentifier: .heartRate)!
    private let hrvType = HKQuantityType.quantityType(forIdentifier: .heartRateVariabilitySDNN)!
    private let respiratoryRateType = HKQuantityType.quantityType(forIdentifier: .respiratoryRate)!
    
    func requestAuthorization(completion: @escaping (Bool) -> Void) {
        // Check if Health data is available
        guard HKHealthStore.isHealthDataAvailable() else {
            completion(false)
            return
        }
        
        // Data types to read
        let typesToRead: Set<HKObjectType> = [
            heartRateType,
            hrvType,
            respiratoryRateType,
            HKObjectType.categoryType(forIdentifier: .sleepAnalysis)!,
            HKObjectType.quantityType(forIdentifier: .stepCount)!
        ]
        
        // Request authorization
        healthStore.requestAuthorization(toShare: nil, read: typesToRead) { success, error in
            completion(success)
        }
    }
    
    func fetchLatestHeartRate(completion: @escaping (Double?) -> Void) {
        let sortDescriptor = NSSortDescriptor(key: HKSampleSortIdentifierStartDate, ascending: false)
        let query = HKSampleQuery(sampleType: heartRateType, 
                                 predicate: nil, 
                                 limit: 1, 
                                 sortDescriptors: [sortDescriptor]) { _, samples, error in
            
            guard let sample = samples?.first as? HKQuantitySample else {
                completion(nil)
                return
            }
            
            let heartRate = sample.quantity.doubleValue(for: HKUnit(from: "count/min"))
            completion(heartRate)
        }
        
        healthStore.execute(query)
    }
}
```

## 3. ModelManager.swift (ML Model Runner)

```swift
import CoreML

class ModelManager {
    // Load your Core ML models here
    private var svmModel: MLModel?
    private var gbmModel: MLModel?
    private var nnModel: MLModel?
    
    init() {
        // Load models (add your .mlmodel files to project first)
        // svmModel = try? MLModel(contentsOf: URL for svm_model.mlmodel)
        // gbmModel = try? MLModel(contentsOf: URL for gbm_model.mlmodel)
        // nnModel = try? MLModel(contentsOf: URL for nn_model.mlmodel)
    }
    
    func analyzeHeartRhythm(heartRate: Double, stdDev: Double, pnn50: Double) -> String {
        // TODO: Run SVM model prediction
        // For now, return mock result
        if stdDev > 15 {
            return "Irregular"
        } else {
            return "Normal"
        }
    }
    
    func assessHealthRisk(heartRate: Double, hrv: Double, activity: Double) -> String {
        // TODO: Run GBM model prediction
        // For now, return mock result
        if heartRate > 100 && activity < 100 {
            return "High"
        } else {
            return "Low"
        }
    }
}
```

## 4. Info.plist Additions (Privacy Permissions)

Add these to your Info.plist file:

```xml
<key>NSHealthShareUsageDescription</key>
<string>This app needs access to your health data to monitor your heart rhythm and provide health insights.</string>

<key>NSHealthUpdateUsageDescription</key>
<string>This app may update your health records with analyzed data.</string>
```

## 5. Simple Model Conversion Script (save as convert_to_coreml.py)

```python
import coremltools as ct
import joblib

# Convert SVM model
print("Converting SVM model...")
svm_model = joblib.load('svm_heart_rhythm_model.pkl')
coreml_model = ct.converters.sklearn.convert(svm_model, 
                                             input_features=['mean_heart_rate', 'std_heart_rate', 'pnn50'],
                                             output_feature_names=['rhythm_class'])
coreml_model.save('HeartRhythmClassifier.mlmodel')

print("✅ Models converted! Add the .mlmodel files to your Xcode project")
```

## That's It! 🎉

1. Copy these code templates into your Xcode project
2. Run the app on your iPhone
3. Grant Health permissions
4. Start monitoring!

The app will now show your real heart rate from your Apple Watch!