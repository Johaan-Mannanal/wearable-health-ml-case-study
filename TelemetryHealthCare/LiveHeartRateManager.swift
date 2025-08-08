//
//  LiveHeartRateManager.swift
//  Rhythm 360
//
//  Real-time heart rate monitoring with streaming architecture
//

import Foundation
import HealthKit
import Combine
import UserNotifications

class LiveHeartRateManager: ObservableObject {
    static let shared = LiveHeartRateManager()
    
    // MARK: - Published Properties
    @Published var currentHeartRate: Double = 0
    @Published var isLiveMonitoring = false
    @Published var recentReadings: [HeartRateReading] = []
    @Published var currentZone: HeartRateZone = .resting
    @Published var isWorkoutActive = false
    @Published var alertStatus: AlertStatus = .normal
    
    // MARK: - Private Properties
    private let healthStore = HKHealthStore()
    private var observerQuery: HKObserverQuery?
    private var anchoredQuery: HKAnchoredObjectQuery?
    private var workoutSession: HKWorkoutSession?
    private var lastAnchor: HKQueryAnchor?
    private let bufferSize = 200 // Circular buffer for live visualization
    private var analysisTimer: Timer?
    private var cancellables = Set<AnyCancellable>()
    
    // ML Analysis Windows
    private var recentWindow: [Double] = [] // Last 5 minutes for ML
    private let windowSize = 300 // 5 minutes at 1 reading/second
    
    // Alert thresholds
    private let restingHighThreshold = 100.0
    private let restingLowThreshold = 50.0
    private let exerciseHighThreshold = 180.0
    
    private init() {
        setupNotifications()
    }
    
    // MARK: - Live Monitoring Setup
    
    func startLiveMonitoring() {
        guard HKHealthStore.isHealthDataAvailable() else { return }
        
        isLiveMonitoring = true
        setupObserverQuery()
        setupAnchoredQuery()
        startAnalysisTimer()
        
        print("🎯 Live heart rate monitoring started")
    }
    
    func stopLiveMonitoring() {
        isLiveMonitoring = false
        
        if let query = observerQuery {
            healthStore.stop(query)
            observerQuery = nil
        }
        if let query = anchoredQuery {
            healthStore.stop(query)
            anchoredQuery = nil
        }
        
        analysisTimer?.invalidate()
        analysisTimer = nil
        
        // Clear buffers to free memory
        recentReadings.removeAll()
        recentWindow.removeAll()
        
        print("⏹ Live heart rate monitoring stopped")
    }
    
    // MARK: - Real-Time Streaming with HKObserverQuery
    
    private func setupObserverQuery() {
        guard let heartRateType = HKQuantityType.quantityType(forIdentifier: .heartRate) else { return }
        
        // Observer query triggers immediately when new data is available
        observerQuery = HKObserverQuery(sampleType: heartRateType, predicate: nil) { [weak self] query, completionHandler, error in
            if let error = error {
                print("❌ Observer query error: \(error.localizedDescription)")
                completionHandler()
                return
            }
            
            // New data available - fetch it with anchored query
            DispatchQueue.main.async {
                self?.fetchNewHeartRateData()
            }
            
            completionHandler()
        }
        
        if let query = observerQuery {
            healthStore.execute(query)
        }
    }
    
    // MARK: - Efficient Data Fetching with HKAnchoredObjectQuery
    
    private func setupAnchoredQuery() {
        guard let heartRateType = HKQuantityType.quantityType(forIdentifier: .heartRate) else { return }
        
        // Only fetch new samples since last anchor
        let query = HKAnchoredObjectQuery(
            type: heartRateType,
            predicate: nil,
            anchor: lastAnchor,
            limit: HKObjectQueryNoLimit
        ) { [weak self] query, samples, deletedObjects, newAnchor, error in
            self?.processNewHeartRateData(samples: samples, anchor: newAnchor)
        }
        
        // Update handler for continuous streaming
        query.updateHandler = { [weak self] query, samples, deletedObjects, newAnchor, error in
            self?.processNewHeartRateData(samples: samples, anchor: newAnchor)
        }
        
        anchoredQuery = query
        healthStore.execute(query)
    }
    
    private func fetchNewHeartRateData() {
        // Triggered by observer query - anchored query will handle it
    }
    
    private func processNewHeartRateData(samples: [HKSample]?, anchor: HKQueryAnchor?) {
        guard let heartRateSamples = samples as? [HKQuantitySample] else { return }
        
        // Update anchor for next query
        if let newAnchor = anchor {
            lastAnchor = newAnchor
        }
        
        DispatchQueue.main.async { [weak self] in
            guard let self = self else { return }
            
            for sample in heartRateSamples {
                let heartRate = sample.quantity.doubleValue(for: HKUnit(from: "count/min"))
                let reading = HeartRateReading(
                    value: heartRate,
                    timestamp: sample.endDate,
                    isWorkout: self.isWorkoutActive
                )
                
                // Update current heart rate
                self.currentHeartRate = heartRate
                
                // Add to circular buffer
                self.recentReadings.append(reading)
                if self.recentReadings.count > self.bufferSize {
                    self.recentReadings.removeFirst()
                }
                
                // Add to ML window
                self.recentWindow.append(heartRate)
                if self.recentWindow.count > self.windowSize {
                    self.recentWindow.removeFirst()
                }
                
                // Update zone and check alerts
                self.updateHeartRateZone(heartRate)
                self.checkForAlerts(heartRate)
            }
            
            #if DEBUG
            print("💗 Live HR update received")
            #endif
        }
    }
    
    // MARK: - Intelligent Workout Detection
    
    func startWorkoutSession(activityType: HKWorkoutActivityType = .other) {
        let configuration = HKWorkoutConfiguration()
        configuration.activityType = activityType
        configuration.locationType = .unknown
        
        do {
            workoutSession = try HKWorkoutSession(healthStore: healthStore, configuration: configuration)
            workoutSession?.startActivity(with: Date())
            isWorkoutActive = true
            
            print("🏃 Workout session started - Enhanced monitoring active")
        } catch {
            print("❌ Failed to start workout session: \(error)")
        }
    }
    
    func endWorkoutSession() {
        workoutSession?.end()
        workoutSession = nil
        isWorkoutActive = false
        
        print("🏁 Workout session ended")
    }
    
    // MARK: - Heart Rate Zones
    
    private func updateHeartRateZone(_ heartRate: Double) {
        if isWorkoutActive {
            currentZone = calculateExerciseZone(heartRate)
        } else {
            currentZone = calculateRestingZone(heartRate)
        }
    }
    
    private func calculateRestingZone(_ heartRate: Double) -> HeartRateZone {
        switch heartRate {
        case 0..<50: return .low
        case 50..<60: return .resting
        case 60..<100: return .normal
        case 100..<120: return .elevated
        default: return .high
        }
    }
    
    private func calculateExerciseZone(_ heartRate: Double) -> HeartRateZone {
        // Using age-based max HR (220 - age), assuming age 30 for demo
        let maxHR = 190.0
        let percentage = (heartRate / maxHR) * 100
        
        switch percentage {
        case 0..<50: return .resting
        case 50..<60: return .warmup
        case 60..<70: return .fatBurn
        case 70..<80: return .cardio
        case 80..<90: return .peak
        default: return .maximum
        }
    }
    
    // MARK: - Smart Alert System
    
    private func checkForAlerts(_ heartRate: Double) {
        let previousStatus = alertStatus
        
        if isWorkoutActive {
            // Exercise thresholds
            if heartRate > exerciseHighThreshold {
                alertStatus = .warning
                if previousStatus != .warning {
                    triggerNotification(
                        title: "High Heart Rate During Exercise",
                        body: "Your heart rate is \(Int(heartRate)) bpm. Consider reducing intensity.",
                        urgency: .medium
                    )
                }
            } else {
                alertStatus = .normal
            }
        } else {
            // Resting thresholds
            if heartRate > restingHighThreshold {
                alertStatus = .critical
                if previousStatus != .critical {
                    triggerNotification(
                        title: "⚠️ High Resting Heart Rate",
                        body: "Your resting heart rate is \(Int(heartRate)) bpm. This may require attention.",
                        urgency: .high
                    )
                }
            } else if heartRate < restingLowThreshold && heartRate > 0 {
                alertStatus = .warning
                if previousStatus != .warning {
                    triggerNotification(
                        title: "Low Heart Rate Detected",
                        body: "Your heart rate is \(Int(heartRate)) bpm. Monitor for symptoms.",
                        urgency: .medium
                    )
                }
            } else {
                alertStatus = .normal
            }
        }
        
        // Check for irregular patterns
        if recentWindow.count >= 10 {
            checkForIrregularities()
        }
    }
    
    private func checkForIrregularities() {
        let recent = Array(recentWindow.suffix(10))
        let mean = recent.reduce(0, +) / Double(recent.count)
        let variance = recent.map { pow($0 - mean, 2) }.reduce(0, +) / Double(recent.count)
        let stdDev = sqrt(variance)
        
        // High variability might indicate irregularity
        if stdDev > 15 && !isWorkoutActive {
            if alertStatus == .normal {
                alertStatus = .monitoring
                triggerNotification(
                    title: "Irregular Heart Rhythm Detected",
                    body: "Your heart rhythm appears irregular. Opening live monitor.",
                    urgency: .medium
                )
            }
        }
    }
    
    // MARK: - ML Analysis Timer
    
    private func startAnalysisTimer() {
        // Run ML analysis every minute on 5-minute windows
        analysisTimer = Timer.scheduledTimer(withTimeInterval: 60, repeats: true) { [weak self] _ in
            self?.runMLAnalysis()
        }
    }
    
    private func runMLAnalysis() {
        guard recentWindow.count >= 60 else { return } // Need at least 1 minute of data
        
        // Prepare data for ML models
        let features = calculateMLFeatures()
        
        // Run analysis in background
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            // This would call your ML models
            // For now, using simplified analysis
            let riskScore = self?.calculateRiskScore(features: features) ?? 0
            
            DispatchQueue.main.async {
                if riskScore > 0.7 {
                    self?.alertStatus = .critical
                    self?.triggerNotification(
                        title: "⚠️ Health Risk Detected",
                        body: "ML analysis detected potential health risk. Please review your data.",
                        urgency: .high
                    )
                }
            }
        }
    }
    
    private func calculateMLFeatures() -> MLFeatures {
        let mean = recentWindow.reduce(0, +) / Double(recentWindow.count)
        let variance = recentWindow.map { pow($0 - mean, 2) }.reduce(0, +) / Double(recentWindow.count)
        let std = sqrt(variance)
        
        // Calculate pNN50
        let differences = zip(recentWindow.dropFirst(), recentWindow).map { abs($0 - $1) }
        let nn50 = differences.filter { $0 > 50 / 60 }.count // Convert to bpm difference
        let pnn50 = Double(nn50) / Double(differences.count)
        
        return MLFeatures(mean: mean, std: std, pnn50: pnn50)
    }
    
    private func calculateRiskScore(features: MLFeatures) -> Double {
        // Simplified risk calculation
        var score = 0.0
        
        if features.mean > 100 && !isWorkoutActive { score += 0.3 }
        if features.std > 20 { score += 0.3 }
        if features.pnn50 < 0.05 { score += 0.4 }
        
        return min(score, 1.0)
    }
    
    // MARK: - Notifications
    
    private func setupNotifications() {
        UNUserNotificationCenter.current().requestAuthorization(options: [.alert, .sound, .badge]) { granted, error in
            if granted {
                print("✅ Notification permission granted")
            }
        }
    }
    
    private func triggerNotification(title: String, body: String, urgency: NotificationUrgency) {
        let content = UNMutableNotificationContent()
        content.title = title
        content.body = body
        content.sound = urgency == .high ? .defaultCritical : .default
        
        // Deep link to live monitoring view
        content.userInfo = ["action": "openLiveMonitor"]
        
        let request = UNNotificationRequest(
            identifier: UUID().uuidString,
            content: content,
            trigger: nil // Immediate
        )
        
        UNUserNotificationCenter.current().add(request)
    }
}

// MARK: - Supporting Types

struct HeartRateReading: Identifiable {
    let id = UUID()
    let value: Double
    let timestamp: Date
    let isWorkout: Bool
}

enum HeartRateZone: String, CaseIterable {
    case low = "Low"
    case resting = "Resting"
    case normal = "Normal"
    case elevated = "Elevated"
    case high = "High"
    case warmup = "Warm Up"
    case fatBurn = "Fat Burn"
    case cardio = "Cardio"
    case peak = "Peak"
    case maximum = "Maximum"
    
    var color: Color {
        switch self {
        case .low: return .blue
        case .resting: return .green
        case .normal: return .green
        case .elevated: return .yellow
        case .high: return .orange
        case .warmup: return .cyan
        case .fatBurn: return .green
        case .cardio: return .yellow
        case .peak: return .orange
        case .maximum: return .red
        }
    }
}

enum AlertStatus {
    case normal
    case monitoring
    case warning
    case critical
}

enum NotificationUrgency {
    case low
    case medium
    case high
}

struct MLFeatures {
    let mean: Double
    let std: Double
    let pnn50: Double
}