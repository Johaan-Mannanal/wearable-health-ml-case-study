//
//  DataManager.swift
//  Rhythm 360
//
//  Core Data manager for persistent health data storage
//

import Foundation
import CoreData
import SwiftUI

class DataManager: ObservableObject {
    static let shared = DataManager()
    
    @Published var healthRecords: [HealthRecord] = []
    
    lazy var persistentContainer: NSPersistentContainer = {
        let container = NSPersistentContainer(name: "HealthDataModel")
        
        // Enable encryption for Core Data
        let storeDescription = container.persistentStoreDescriptions.first
        storeDescription?.setOption(FileProtectionType.complete as NSObject,
                                   forKey: NSPersistentStoreFileProtectionKey)
        
        container.loadPersistentStores { _, error in
            if let error = error {
                print("❌ Core Data failed to load: \(error.localizedDescription)")
            } else {
                print("✅ Core Data loaded successfully with encryption")
            }
        }
        return container
    }()
    
    private init() {
        fetchHealthRecords()
    }
    
    // MARK: - Core Data Operations
    
    func save() {
        let context = persistentContainer.viewContext
        
        if context.hasChanges {
            do {
                try context.save()
                print("✅ Core Data: Changes saved")
            } catch {
                print("❌ Core Data save error: \(error.localizedDescription)")
            }
        }
    }
    
    func saveHealthAssessment(_ assessment: HealthAssessment, healthData: HealthKitData) {
        let context = persistentContainer.viewContext
        let record = HealthRecord(context: context)
        
        // Set all properties
        record.id = UUID()
        record.date = Date()
        record.heartRate = healthData.meanHeartRate
        record.heartRateStd = healthData.stdHeartRate
        record.pnn50 = healthData.pnn50
        record.hrvMean = healthData.hrvMean
        record.respiratoryRate = healthData.respiratoryRate
        record.activityLevel = healthData.activityLevel
        record.sleepQuality = healthData.sleepQuality
        
        // Assessment results
        record.overallStatus = assessment.overallStatus
        record.rhythmStatus = assessment.rhythmStatus
        record.rhythmConfidence = assessment.rhythmConfidence
        record.riskLevel = assessment.riskLevel
        record.riskConfidence = assessment.riskConfidence
        record.hrvPattern = assessment.hrvPattern
        record.patternConfidence = assessment.patternConfidence
        
        save()
        fetchHealthRecords()
        print("✅ Health assessment saved to Core Data")
    }
    
    func fetchHealthRecords() {
        let request: NSFetchRequest<HealthRecord> = HealthRecord.fetchRequest()
        request.sortDescriptors = [NSSortDescriptor(key: "date", ascending: false)]
        
        do {
            healthRecords = try persistentContainer.viewContext.fetch(request)
            print("✅ Fetched \(healthRecords.count) health records")
        } catch {
            print("❌ Error fetching health records: \(error)")
        }
    }
    
    func fetchRecords(for days: Int) -> [HealthRecord] {
        let request: NSFetchRequest<HealthRecord> = HealthRecord.fetchRequest()
        let startDate = Calendar.current.date(byAdding: .day, value: -days, to: Date())!
        request.predicate = NSPredicate(format: "date >= %@", startDate as NSDate)
        request.sortDescriptors = [NSSortDescriptor(key: "date", ascending: true)]
        
        do {
            return try persistentContainer.viewContext.fetch(request)
        } catch {
            print("❌ Error fetching records for \(days) days: \(error)")
            return []
        }
    }
    
    func deleteRecord(_ record: HealthRecord) {
        persistentContainer.viewContext.delete(record)
        save()
        fetchHealthRecords()
    }
    
    func deleteAllRecords() {
        let request: NSFetchRequest<NSFetchRequestResult> = HealthRecord.fetchRequest()
        let deleteRequest = NSBatchDeleteRequest(fetchRequest: request)
        
        do {
            try persistentContainer.viewContext.execute(deleteRequest)
            save()
            fetchHealthRecords()
            print("✅ All health records deleted")
        } catch {
            print("❌ Error deleting all records: \(error)")
        }
    }
    
    // MARK: - Statistics
    
    func getAverageHeartRate(days: Int = 7) -> Double {
        let records = fetchRecords(for: days)
        guard !records.isEmpty else { return 0 }
        return records.map { $0.heartRate }.reduce(0, +) / Double(records.count)
    }
    
    func getHealthTrends(days: Int = 7) -> HealthTrends {
        let records = fetchRecords(for: days)
        guard !records.isEmpty else {
            return HealthTrends(
                averageHeartRate: 0,
                averageHRV: 0,
                averageRespiratoryRate: 0,
                totalActivity: 0,
                averageSleepQuality: 0,
                riskTrend: .stable,
                recordCount: 0
            )
        }
        
        let avgHR = records.map { $0.heartRate }.reduce(0, +) / Double(records.count)
        let avgHRV = records.map { $0.hrvMean }.reduce(0, +) / Double(records.count)
        let avgRR = records.map { $0.respiratoryRate }.reduce(0, +) / Double(records.count)
        let totalActivity = records.map { $0.activityLevel }.reduce(0, +)
        let avgSleep = records.map { $0.sleepQuality }.reduce(0, +) / Double(records.count)
        
        // Determine risk trend
        let riskTrend: RiskTrend
        if records.count > 1 {
            let recentRisks = records.suffix(3).compactMap { $0.riskLevel == "High" ? 1.0 : 0.0 }
            let olderRisks = records.prefix(records.count - 3).compactMap { $0.riskLevel == "High" ? 1.0 : 0.0 }
            
            let recentAvg = recentRisks.isEmpty ? 0 : recentRisks.reduce(0, +) / Double(recentRisks.count)
            let olderAvg = olderRisks.isEmpty ? 0 : olderRisks.reduce(0, +) / Double(olderRisks.count)
            
            if recentAvg > olderAvg + 0.2 {
                riskTrend = .worsening
            } else if recentAvg < olderAvg - 0.2 {
                riskTrend = .improving
            } else {
                riskTrend = .stable
            }
        } else {
            riskTrend = .stable
        }
        
        return HealthTrends(
            averageHeartRate: avgHR,
            averageHRV: avgHRV,
            averageRespiratoryRate: avgRR,
            totalActivity: totalActivity,
            averageSleepQuality: avgSleep,
            riskTrend: riskTrend,
            recordCount: records.count
        )
    }
}

// MARK: - Core Data Entity Model

/// Core Data entity representing a single health assessment record
/// 
/// This NSManagedObject subclass stores comprehensive health data including
/// raw sensor measurements and processed ML model results. Each record
/// represents a complete health assessment at a specific point in time.
/// 
/// - Note: Conforms to Identifiable for SwiftUI List integration
/// - Important: All properties are automatically managed by Core Data
@objc(HealthRecord)
public class HealthRecord: NSManagedObject, Identifiable {
    // Core Data properties are defined in the .xcdatamodeld file
    // and generated automatically by Xcode
}

// MARK: - Core Data Fetch Request Factory

extension HealthRecord {
    
    /// Creates a type-safe fetch request for HealthRecord entities
    /// 
    /// This factory method provides a convenient way to create fetch requests
    /// with proper type safety and entity name validation.
    /// 
    /// - Returns: Configured NSFetchRequest for HealthRecord entities
    /// - Note: Used internally by DataManager for all query operations
    @nonobjc public class func fetchRequest() -> NSFetchRequest<HealthRecord> {
        return NSFetchRequest<HealthRecord>(entityName: "HealthRecord")
    }
    
    // MARK: - Record Metadata
    /// Unique identifier for the health record
    @NSManaged public var id: UUID?
    
    /// Timestamp when the assessment was performed
    @NSManaged public var date: Date?
    
    // MARK: - Raw Health Data
    /// Mean heart rate in beats per minute
    @NSManaged public var heartRate: Double
    
    /// Standard deviation of heart rate (variability measure)
    @NSManaged public var heartRateStd: Double
    
    /// Percentage of successive RR intervals differing by >50ms
    @NSManaged public var pnn50: Double
    
    /// Mean heart rate variability in milliseconds
    @NSManaged public var hrvMean: Double
    
    /// Respiratory rate in breaths per minute
    @NSManaged public var respiratoryRate: Double
    
    /// Daily active energy expenditure in kilocalories
    @NSManaged public var activityLevel: Double
    
    /// Sleep efficiency score (0.0-1.0)
    @NSManaged public var sleepQuality: Double
    
    // MARK: - ML Model Assessment Results
    /// Aggregated health status from all models
    @NSManaged public var overallStatus: String?
    
    /// SVM heart rhythm classification result
    @NSManaged public var rhythmStatus: String?
    
    /// Confidence score for rhythm classification (0.0-1.0)
    @NSManaged public var rhythmConfidence: Double
    
    /// XGBoost health risk assessment result
    @NSManaged public var riskLevel: String?
    
    /// Confidence score for risk assessment (0.0-1.0)
    @NSManaged public var riskConfidence: Double
    
    /// Neural network HRV pattern classification result
    @NSManaged public var hrvPattern: String?
    
    /// Confidence score for HRV pattern classification (0.0-1.0)
    @NSManaged public var patternConfidence: Double
}

// MARK: - Health Analytics Data Models

/// Comprehensive health trends analysis over a specified time period
/// 
/// This structure contains statistical analysis of health metrics over time,
/// providing insights into health patterns, improvements, and concerning trends.
/// 
/// - Note: Used for trend visualization and long-term health monitoring
struct HealthTrends: Codable {
    /// Average heart rate over the analysis period (BPM)
    let averageHeartRate: Double
    
    /// Average heart rate variability over the analysis period (ms)
    let averageHRV: Double
    
    /// Average respiratory rate over the analysis period (breaths/min)
    let averageRespiratoryRate: Double
    
    /// Total cumulative activity energy over the analysis period (kcal)
    let totalActivity: Double
    
    /// Average sleep efficiency over the analysis period (0.0-1.0)
    let averageSleepQuality: Double
    
    /// Risk trend direction over the analysis period
    let riskTrend: RiskTrend
    
    /// Number of data points included in the analysis
    let recordCount: Int
}

/// Enumeration representing the direction of health risk trends over time
/// 
/// Used to indicate whether a user's cardiovascular risk is improving,
/// remaining stable, or worsening based on recent health assessments.
/// 
/// - Note: Calculated by comparing recent high-risk assessment frequency
///         to historical baseline over the analysis period
enum RiskTrend: String, Codable {
    /// Health risk is decreasing (fewer high-risk assessments recently)
    case improving
    
    /// Health risk remains consistent (no significant change)
    case stable
    
    /// Health risk is increasing (more high-risk assessments recently)
    case worsening
}