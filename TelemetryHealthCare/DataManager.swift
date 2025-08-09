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

// MARK: - Core Data Model

@objc(HealthRecord)
public class HealthRecord: NSManagedObject, Identifiable {
    
}

extension HealthRecord {
    @nonobjc public class func fetchRequest() -> NSFetchRequest<HealthRecord> {
        return NSFetchRequest<HealthRecord>(entityName: "HealthRecord")
    }
    
    @NSManaged public var id: UUID?
    @NSManaged public var date: Date?
    @NSManaged public var heartRate: Double
    @NSManaged public var heartRateStd: Double
    @NSManaged public var pnn50: Double
    @NSManaged public var hrvMean: Double
    @NSManaged public var respiratoryRate: Double
    @NSManaged public var activityLevel: Double
    @NSManaged public var sleepQuality: Double
    @NSManaged public var overallStatus: String?
    @NSManaged public var rhythmStatus: String?
    @NSManaged public var rhythmConfidence: Double
    @NSManaged public var riskLevel: String?
    @NSManaged public var riskConfidence: Double
    @NSManaged public var hrvPattern: String?
    @NSManaged public var patternConfidence: Double
}

// MARK: - Supporting Types

struct HealthTrends: Codable {
    let averageHeartRate: Double
    let averageHRV: Double
    let averageRespiratoryRate: Double
    let totalActivity: Double
    let averageSleepQuality: Double
    let riskTrend: RiskTrend
    let recordCount: Int
}

enum RiskTrend: String, Codable {
    case improving
    case stable
    case worsening
}