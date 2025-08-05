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
        let typesToRead = Set([heartRateType, hrvType, bpSystolicType, bpDiastolicType])

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
            completion(nil)
            return
        }

        // Get HRV data from the last hour
        let endDate = Date()
        let startDate = Date(timeIntervalSinceNow: -3600) // 1 hour ago
        let predicate = HKQuery.predicateForSamples(withStart: startDate, end: endDate, options: .strictStartDate)
        
        let sort = NSSortDescriptor(key: HKSampleSortIdentifierEndDate, ascending: false)
        let query = HKSampleQuery(sampleType: hrvType,
                                  predicate: predicate,
                                  limit: 50,
                                  sortDescriptors: [sort]) { (query, samples, error) in
            guard let samples = samples as? [HKQuantitySample], error == nil else {
                completion(nil)
                return
            }

            let hrvData = samples.map { ($0.quantity.doubleValue(for: HKUnit.secondUnit(with: .milli)), $0.endDate) }
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
}
