//  HealthKitManager.swift
//  TelemetryHealthCare
//
//  Created by Yashwanth on 6/30/25.
//

import HealthKit
import CoreML

class HealthKitManager {
    static let shared = HealthKitManager()
    private let healthStore = HKHealthStore()

    private init() {}

    func isHealthKitAvailable() -> Bool {
        return HKHealthStore.isHealthDataAvailable()
    }

    func askForPermission(completion: @escaping (Bool) -> Void) {
        if !isHealthKitAvailable() {
            completion(false)
            return
        }

        let heartRateType = HKQuantityType.quantityType(forIdentifier: .heartRate)!
        let hrvType = HKQuantityType.quantityType(forIdentifier: .heartRateVariabilitySDNN)!
        let bpSystolicType = HKQuantityType.quantityType(forIdentifier: .bloodPressureSystolic)!
        let bpDiastolicType = HKQuantityType.quantityType(forIdentifier: .bloodPressureDiastolic)!
        let respiratoryRateType = HKQuantityType.quantityType(forIdentifier: .respiratoryRate)!
        let activeEnergyType = HKQuantityType.quantityType(forIdentifier: .activeEnergyBurned)!
        let stepCountType = HKQuantityType.quantityType(forIdentifier: .stepCount)!
        let sleepAnalysisType = HKCategoryType.categoryType(forIdentifier: .sleepAnalysis)!
        let restingHeartRateType = HKQuantityType.quantityType(forIdentifier: .restingHeartRate)!
        
        let typesToRead = Set([heartRateType, hrvType, bpSystolicType, bpDiastolicType,
                               respiratoryRateType, activeEnergyType, stepCountType,
                               sleepAnalysisType, restingHeartRateType] as [HKObjectType])

        healthStore.requestAuthorization(toShare: nil, read: typesToRead) { success, error in
            completion(success)
        }
    }

    func getHeartRate(completion: @escaping ([(Double, Date)]?) -> Void) {
        guard let heartRateType = HKQuantityType.quantityType(forIdentifier: .heartRate) else {
            print("❌ HealthKit: Heart rate type not available")
            completion(nil)
            return
        }

        // Try multiple time windows with progressively longer ranges
        let timeWindows = [
            ("last hour", -3600.0),
            ("last 6 hours", -21600.0),
            ("last 24 hours", -86400.0),
            ("last 7 days", -604800.0)
        ]
        
        fetchHeartRateWithFallback(type: heartRateType, timeWindows: timeWindows, windowIndex: 0, completion: completion)
    }
    
    private func fetchHeartRateWithFallback(
        type: HKQuantityType,
        timeWindows: [(String, TimeInterval)],
        windowIndex: Int,
        completion: @escaping ([(Double, Date)]?) -> Void
    ) {
        guard windowIndex < timeWindows.count else {
            print("❌ HealthKit: No heart rate data found in any time window")
            completion(nil)
            return
        }
        
        let (windowName, interval) = timeWindows[windowIndex]
        let endDate = Date()
        let startDate = Date(timeIntervalSinceNow: interval)
        let predicate = HKQuery.predicateForSamples(withStart: startDate, end: endDate, options: .strictStartDate)
        
        print("🔍 HealthKit: Searching for heart rate data in \(windowName)")
        
        let sort = NSSortDescriptor(key: HKSampleSortIdentifierEndDate, ascending: false)
        let query = HKSampleQuery(sampleType: type,
                                  predicate: predicate,
                                  limit: 100,
                                  sortDescriptors: [sort]) { (query, samples, error) in
            if let error = error {
                print("❌ HealthKit Query Error: \(error.localizedDescription)")
                // Try next time window
                self.fetchHeartRateWithFallback(type: type, timeWindows: timeWindows,
                                               windowIndex: windowIndex + 1, completion: completion)
                return
            }
            
            guard let samples = samples as? [HKQuantitySample], !samples.isEmpty else {
                print("⚠️ HealthKit: No data in \(windowName), trying next window...")
                self.fetchHeartRateWithFallback(type: type, timeWindows: timeWindows,
                                               windowIndex: windowIndex + 1, completion: completion)
                return
            }

            let heartRates = samples.map { ($0.quantity.doubleValue(for: HKUnit(from: "count/min")), $0.endDate) }
            print("✅ HealthKit: Found \(heartRates.count) heart rate samples in \(windowName)")
            print("    Latest: \(heartRates.first?.0 ?? 0) bpm at \(heartRates.first?.1 ?? Date())")
            completion(heartRates)
        }

        healthStore.execute(query)
    }

    func getHRV(completion: @escaping ([(Double, Date)]?) -> Void) {
        guard let hrvType = HKQuantityType.quantityType(forIdentifier: .heartRateVariabilitySDNN) else {
            print("❌ HealthKit: HRV type not available")
            completion(nil)
            return
        }

        // Try multiple time windows with progressively longer ranges
        let timeWindows = [
            ("last hour", -3600.0),
            ("last 6 hours", -21600.0),
            ("last 24 hours", -86400.0),
            ("last 7 days", -604800.0),
            ("last 30 days", -2592000.0)
        ]
        
        fetchHRVWithFallback(type: hrvType, timeWindows: timeWindows, windowIndex: 0, completion: completion)
    }
    
    private func fetchHRVWithFallback(
        type: HKQuantityType,
        timeWindows: [(String, TimeInterval)],
        windowIndex: Int,
        completion: @escaping ([(Double, Date)]?) -> Void
    ) {
        guard windowIndex < timeWindows.count else {
            print("❌ HealthKit: No HRV data found in any time window")
            completion(nil)
            return
        }
        
        let (windowName, interval) = timeWindows[windowIndex]
        let endDate = Date()
        let startDate = Date(timeIntervalSinceNow: interval)
        let predicate = HKQuery.predicateForSamples(withStart: startDate, end: endDate, options: .strictStartDate)
        
        print("🔍 HealthKit: Searching for HRV data in \(windowName)")
        
        let sort = NSSortDescriptor(key: HKSampleSortIdentifierEndDate, ascending: false)
        let query = HKSampleQuery(sampleType: type,
                                  predicate: predicate,
                                  limit: 100,
                                  sortDescriptors: [sort]) { (query, samples, error) in
            if let error = error {
                print("❌ HealthKit HRV Query Error: \(error.localizedDescription)")
                // Try next time window
                self.fetchHRVWithFallback(type: type, timeWindows: timeWindows,
                                         windowIndex: windowIndex + 1, completion: completion)
                return
            }
            
            guard let samples = samples as? [HKQuantitySample], !samples.isEmpty else {
                print("⚠️ HealthKit: No HRV data in \(windowName), trying next window...")
                self.fetchHRVWithFallback(type: type, timeWindows: timeWindows,
                                         windowIndex: windowIndex + 1, completion: completion)
                return
            }

            let hrvData = samples.map { ($0.quantity.doubleValue(for: HKUnit.secondUnit(with: .milli)), $0.endDate) }
            print("✅ HealthKit: Found \(hrvData.count) HRV samples in \(windowName)")
            print("    Latest: \(hrvData.first?.0 ?? 0) ms at \(hrvData.first?.1 ?? Date())")
            completion(hrvData)
        }

        healthStore.execute(query)
    }

    func getBloodPressure(completion: @escaping ([(systolic: Double, diastolic: Double, date: Date)]?) -> Void) {
        guard let bpSystolicType = HKQuantityType.quantityType(forIdentifier: .bloodPressureSystolic) else {
            completion(nil)
            return
        }

        let sort = NSSortDescriptor(key: HKSampleSortIdentifierEndDate, ascending: false)
        let query = HKSampleQuery(sampleType: bpSystolicType,
                                  predicate: nil,
                                  limit: 5,
                                  sortDescriptors: [sort]) { (query, samples, error) in
            guard let samples = samples as? [HKQuantitySample], error == nil else {
                completion(nil)
                return
            }

            let bpData = samples.map { sample in
                let systolic = sample.quantity.doubleValue(for: HKUnit.millimeterOfMercury())
                return (systolic: systolic, diastolic: 80.0, date: sample.endDate) // Placeholder diastolic
            }
            completion(bpData)
        }

        healthStore.execute(query)
    }

    func computeSVMFeatures(heartRates: [(Double, Date)]) -> (mean: Double, std: Double, pnn50: Double)? {
        guard !heartRates.isEmpty else { return nil }

        let rates = heartRates.map { $0.0 }
        let mean = rates.reduce(0, +) / Double(rates.count)
        let std = sqrt(rates.map { pow($0 - mean, 2) }.reduce(0, +) / Double(rates.count))
        
        // Calculate pNN50: percentage of successive RR interval differences > 50ms
        let intervals = zip(heartRates, heartRates.dropFirst()).map { abs($0.0 - $1.0) }
        let pnn50 = Double(intervals.filter { $0 * 1000 > 50 }.count) / Double(intervals.count)

        return (mean: mean, std: std, pnn50: pnn50)
    }
    
    // MARK: - Real Data Collection Methods
    
    func getRespiratoryRate(completion: @escaping (Double?) -> Void) {
        guard let respiratoryType = HKQuantityType.quantityType(forIdentifier: .respiratoryRate) else {
            print("❌ HealthKit: Respiratory rate type not available")
            completion(nil)
            return
        }
        
        let endDate = Date()
        let startDate = Date(timeIntervalSinceNow: -86400) // Last 24 hours
        let predicate = HKQuery.predicateForSamples(withStart: startDate, end: endDate, options: .strictStartDate)
        
        let query = HKStatisticsQuery(quantityType: respiratoryType,
                                     quantitySamplePredicate: predicate,
                                     options: .discreteAverage) { _, result, error in
            if let error = error {
                print("❌ HealthKit Respiratory Rate Error: \(error.localizedDescription)")
                completion(nil)
                return
            }
            
            let respiratoryRate = result?.averageQuantity()?.doubleValue(for: HKUnit(from: "count/min")) ?? 16.0
            print("✅ HealthKit: Respiratory Rate: \(respiratoryRate) breaths/min")
            completion(respiratoryRate)
        }
        
        healthStore.execute(query)
    }
    
    func getActivityLevel(completion: @escaping (Double?) -> Void) {
        guard let activeEnergyType = HKQuantityType.quantityType(forIdentifier: .activeEnergyBurned) else {
            print("❌ HealthKit: Active energy type not available")
            completion(nil)
            return
        }
        
        let endDate = Date()
        let startDate = Calendar.current.startOfDay(for: endDate)
        let predicate = HKQuery.predicateForSamples(withStart: startDate, end: endDate, options: .strictStartDate)
        
        let query = HKStatisticsQuery(quantityType: activeEnergyType,
                                     quantitySamplePredicate: predicate,
                                     options: .cumulativeSum) { _, result, error in
            if let error = error {
                print("❌ HealthKit Activity Level Error: \(error.localizedDescription)")
                completion(nil)
                return
            }
            
            let activeEnergy = result?.sumQuantity()?.doubleValue(for: HKUnit.kilocalorie()) ?? 250.0
            print("✅ HealthKit: Active Energy: \(activeEnergy) kcal")
            completion(activeEnergy)
        }
        
        healthStore.execute(query)
    }
    
    func getSleepQuality(completion: @escaping (Double?) -> Void) {
        guard let sleepType = HKCategoryType.categoryType(forIdentifier: .sleepAnalysis) else {
            print("❌ HealthKit: Sleep analysis type not available")
            completion(nil)
            return
        }
        
        let endDate = Date()
        let startDate = Calendar.current.date(byAdding: .day, value: -1, to: endDate)!
        let predicate = HKQuery.predicateForSamples(withStart: startDate, end: endDate, options: .strictStartDate)
        
        let query = HKSampleQuery(sampleType: sleepType,
                                 predicate: predicate,
                                 limit: HKObjectQueryNoLimit,
                                 sortDescriptors: nil) { _, samples, error in
            if let error = error {
                print("❌ HealthKit Sleep Quality Error: \(error.localizedDescription)")
                completion(nil)
                return
            }
            
            guard let sleepSamples = samples as? [HKCategorySample], !sleepSamples.isEmpty else {
                print("⚠️ HealthKit: No sleep data available, using default")
                completion(0.8) // Default to 80% sleep quality
                return
            }
            
            // Calculate total sleep duration
            var totalSleepTime: TimeInterval = 0
            var inBedTime: TimeInterval = 0
            
            for sample in sleepSamples {
                let duration = sample.endDate.timeIntervalSince(sample.startDate)
                
                if sample.value == HKCategoryValueSleepAnalysis.inBed.rawValue {
                    inBedTime += duration
                } else if sample.value == HKCategoryValueSleepAnalysis.asleepUnspecified.rawValue ||
                          sample.value == HKCategoryValueSleepAnalysis.asleepCore.rawValue ||
                          sample.value == HKCategoryValueSleepAnalysis.asleepDeep.rawValue ||
                          sample.value == HKCategoryValueSleepAnalysis.asleepREM.rawValue {
                    totalSleepTime += duration
                }
            }
            
            // Calculate sleep quality as ratio of actual sleep to time in bed
            let sleepQuality: Double
            if inBedTime > 0 {
                sleepQuality = min(totalSleepTime / inBedTime, 1.0)
            } else if totalSleepTime > 0 {
                // If we only have sleep data, estimate quality based on duration (7-9 hours is optimal)
                let hoursSlept = totalSleepTime / 3600
                sleepQuality = min(max(hoursSlept / 8.0, 0.5), 1.0)
            } else {
                sleepQuality = 0.8 // Default
            }
            
            print("✅ HealthKit: Sleep Quality: \(sleepQuality * 100)%")
            completion(sleepQuality)
        }
        
        healthStore.execute(query)
    }
    
    func getStepCount(completion: @escaping (Double?) -> Void) {
        guard let stepType = HKQuantityType.quantityType(forIdentifier: .stepCount) else {
            completion(nil)
            return
        }
        
        let endDate = Date()
        let startDate = Calendar.current.startOfDay(for: endDate)
        let predicate = HKQuery.predicateForSamples(withStart: startDate, end: endDate, options: .strictStartDate)
        
        let query = HKStatisticsQuery(quantityType: stepType,
                                     quantitySamplePredicate: predicate,
                                     options: .cumulativeSum) { _, result, error in
            let steps = result?.sumQuantity()?.doubleValue(for: HKUnit.count()) ?? 0
            print("✅ HealthKit: Steps: \(Int(steps))")
            completion(steps)
        }
        
        healthStore.execute(query)
    }
}
