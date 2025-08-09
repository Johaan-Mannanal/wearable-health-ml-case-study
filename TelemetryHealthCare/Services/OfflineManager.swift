//
//  OfflineManager.swift
//  TelemetryHealthCare
//
//  Created by Assistant on 2025-01-09.
//

import Foundation
import SwiftUI
import Network

// MARK: - Offline Manager
class OfflineManager: ObservableObject {
    static let shared = OfflineManager()
    
    @Published var isOffline = false
    @Published var cachedDataAvailable = false
    @Published var lastSyncTime: Date?
    
    private let monitor = NWPathMonitor()
    private let queue = DispatchQueue(label: "com.rhythm360.networkmonitor")
    
    private var cachedHealthData: CachedHealthData?
    private let cacheExpiration: TimeInterval = 3600 // 1 hour
    
    struct CachedHealthData: Codable {
        let timestamp: Date
        let healthData: HealthKitData
        let assessment: HealthAssessment
        let trends: HealthTrends?
        
        var isExpired: Bool {
            Date().timeIntervalSince(timestamp) > 3600 // 1 hour expiration
        }
    }
    
    private init() {
        setupNetworkMonitoring()
        loadCachedData()
    }
    
    // MARK: - Network Monitoring
    private func setupNetworkMonitoring() {
        monitor.pathUpdateHandler = { [weak self] path in
            DispatchQueue.main.async {
                self?.isOffline = path.status != .satisfied
                
                if path.status == .satisfied && self?.cachedDataAvailable == true {
                    // Network restored, sync data
                    self?.syncOfflineData()
                }
            }
        }
        monitor.start(queue: queue)
    }
    
    // MARK: - Data Caching
    func cacheHealthData(_ healthData: HealthKitData, assessment: HealthAssessment, trends: HealthTrends? = nil) {
        let cached = CachedHealthData(
            timestamp: Date(),
            healthData: healthData,
            assessment: assessment,
            trends: trends
        )
        
        cachedHealthData = cached
        saveCacheToFile(cached)
        
        DispatchQueue.main.async {
            self.cachedDataAvailable = true
            self.lastSyncTime = Date()
        }
    }
    
    func getCachedData() -> (healthData: HealthKitData, assessment: HealthAssessment)? {
        guard let cached = cachedHealthData else {
            loadCachedData()
            guard let loaded = cachedHealthData else { return nil }
            return (loaded.healthData, loaded.assessment)
        }
        
        if cached.isExpired {
            return nil
        }
        
        return (cached.healthData, cached.assessment)
    }
    
    func getCachedTrends() -> HealthTrends? {
        return cachedHealthData?.trends
    }
    
    // MARK: - File Persistence
    private var cacheFileURL: URL {
        let documentsPath = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        return documentsPath.appendingPathComponent("health_cache.json")
    }
    
    private func saveCacheToFile(_ cache: CachedHealthData) {
        do {
            let encoder = JSONEncoder()
            encoder.dateEncodingStrategy = .iso8601
            let data = try encoder.encode(cache)
            try data.write(to: cacheFileURL)
        } catch {
            ErrorManager.shared.handleSilent(
                AppError.storageError("Failed to save offline cache"),
                context: "OfflineManager.saveCacheToFile"
            )
        }
    }
    
    private func loadCachedData() {
        do {
            let data = try Data(contentsOf: cacheFileURL)
            let decoder = JSONDecoder()
            decoder.dateDecodingStrategy = .iso8601
            cachedHealthData = try decoder.decode(CachedHealthData.self, from: data)
            
            DispatchQueue.main.async {
                self.cachedDataAvailable = self.cachedHealthData != nil && !self.cachedHealthData!.isExpired
                self.lastSyncTime = self.cachedHealthData?.timestamp
            }
        } catch {
            // No cache file exists or failed to load - this is okay
            cachedDataAvailable = false
        }
    }
    
    // MARK: - Offline Queue
    private var offlineQueue: [OfflineAction] = []
    
    struct OfflineAction: Codable {
        let id: UUID
        let timestamp: Date
        let type: ActionType
        let data: Data?
        
        enum ActionType: String, Codable {
            case saveAssessment
            case exportData
            case updateSettings
        }
    }
    
    func queueOfflineAction(_ action: OfflineAction) {
        offlineQueue.append(action)
        saveOfflineQueue()
    }
    
    private func saveOfflineQueue() {
        let documentsPath = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        let queuePath = documentsPath.appendingPathComponent("offline_queue.json")
        
        if let data = try? JSONEncoder().encode(offlineQueue) {
            try? data.write(to: queuePath)
        }
    }
    
    private func loadOfflineQueue() {
        let documentsPath = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        let queuePath = documentsPath.appendingPathComponent("offline_queue.json")
        
        if let data = try? Data(contentsOf: queuePath),
           let queue = try? JSONDecoder().decode([OfflineAction].self, from: data) {
            offlineQueue = queue
        }
    }
    
    // MARK: - Data Synchronization
    private func syncOfflineData() {
        guard !offlineQueue.isEmpty else { return }
        
        // Process offline queue
        for action in offlineQueue {
            processOfflineAction(action)
        }
        
        // Clear queue after processing
        offlineQueue.removeAll()
        saveOfflineQueue()
        
        // Update sync time
        DispatchQueue.main.async {
            self.lastSyncTime = Date()
        }
    }
    
    private func processOfflineAction(_ action: OfflineAction) {
        switch action.type {
        case .saveAssessment:
            // In production, this would sync to a backend
            print("Syncing offline assessment from \(action.timestamp)")
        case .exportData:
            print("Processing offline export request")
        case .updateSettings:
            print("Syncing offline settings update")
        }
    }
    
    // MARK: - UI Helpers
    func offlineStatusMessage() -> String {
        if isOffline {
            if cachedDataAvailable {
                return "Offline Mode - Using cached data from \(lastSyncTime?.formatted() ?? "unknown")"
            } else {
                return "Offline - No cached data available"
            }
        } else {
            return "Online - Real-time data"
        }
    }
    
    func offlineStatusColor() -> Color {
        if isOffline {
            return cachedDataAvailable ? .orange : .red
        } else {
            return .green
        }
    }
}

// MARK: - Offline Status View
struct OfflineStatusView: View {
    @ObservedObject var offlineManager = OfflineManager.shared
    
    var body: some View {
        if offlineManager.isOffline {
            HStack(spacing: 8) {
                Image(systemName: "wifi.slash")
                    .foregroundColor(.white)
                    .accessibilityLabel("Offline")
                
                Text(offlineManager.cachedDataAvailable ? "Offline Mode" : "No Connection")
                    .font(.caption)
                    .foregroundColor(.white)
                
                if offlineManager.cachedDataAvailable {
                    Text("(Cached)")
                        .font(.caption2)
                        .foregroundColor(.white.opacity(0.8))
                }
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 6)
            .background(offlineManager.cachedDataAvailable ? Color.orange : Color.red)
            .cornerRadius(20)
            .accessibilityElement(children: .combine)
            .accessibilityLabel(offlineManager.offlineStatusMessage())
        }
    }
}

// MARK: - HealthKitData Extension for Codable
extension HealthKitData: Codable {
    enum CodingKeys: String, CodingKey {
        case meanHeartRate, stdHeartRate, pnn50, hrvMean
        case respiratoryRate, activityLevel, sleepQuality, recentHeartRates
    }
    
    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        meanHeartRate = try container.decode(Double.self, forKey: .meanHeartRate)
        stdHeartRate = try container.decode(Double.self, forKey: .stdHeartRate)
        pnn50 = try container.decode(Double.self, forKey: .pnn50)
        hrvMean = try container.decode(Double.self, forKey: .hrvMean)
        respiratoryRate = try container.decode(Double.self, forKey: .respiratoryRate)
        activityLevel = try container.decode(Double.self, forKey: .activityLevel)
        sleepQuality = try container.decode(Double.self, forKey: .sleepQuality)
        recentHeartRates = try container.decode([Double].self, forKey: .recentHeartRates)
    }
    
    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(meanHeartRate, forKey: .meanHeartRate)
        try container.encode(stdHeartRate, forKey: .stdHeartRate)
        try container.encode(pnn50, forKey: .pnn50)
        try container.encode(hrvMean, forKey: .hrvMean)
        try container.encode(respiratoryRate, forKey: .respiratoryRate)
        try container.encode(activityLevel, forKey: .activityLevel)
        try container.encode(sleepQuality, forKey: .sleepQuality)
        try container.encode(recentHeartRates, forKey: .recentHeartRates)
    }
}

// MARK: - HealthAssessment Extension for Codable
extension HealthAssessment: Codable {
    enum CodingKeys: String, CodingKey {
        case rhythmStatus, rhythmConfidence, riskLevel, riskConfidence
        case hrvPattern, patternConfidence, timestamp
    }
    
    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        rhythmStatus = try container.decode(String.self, forKey: .rhythmStatus)
        rhythmConfidence = try container.decode(Double.self, forKey: .rhythmConfidence)
        riskLevel = try container.decode(String.self, forKey: .riskLevel)
        riskConfidence = try container.decode(Double.self, forKey: .riskConfidence)
        hrvPattern = try container.decode(String.self, forKey: .hrvPattern)
        patternConfidence = try container.decode(Double.self, forKey: .patternConfidence)
        timestamp = try container.decode(Date.self, forKey: .timestamp)
    }
    
    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(rhythmStatus, forKey: .rhythmStatus)
        try container.encode(rhythmConfidence, forKey: .rhythmConfidence)
        try container.encode(riskLevel, forKey: .riskLevel)
        try container.encode(riskConfidence, forKey: .riskConfidence)
        try container.encode(hrvPattern, forKey: .hrvPattern)
        try container.encode(patternConfidence, forKey: .patternConfidence)
        try container.encode(timestamp, forKey: .timestamp)
    }
}