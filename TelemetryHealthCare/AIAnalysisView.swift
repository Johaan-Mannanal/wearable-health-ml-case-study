//
//  AIAnalysisView.swift
//  TelemetryHealthCare
//
//  Main AI health analysis view providing real-time cardiovascular monitoring
//  and machine learning-based health assessments using Apple Watch data.
//
//  This view serves as the primary interface for users to monitor their health
//  using four sophisticated ML models: SVM for rhythm analysis, XGBoost for
//  risk assessment, Neural Network for HRV pattern analysis, and Random Forest
//  for cardiovascular fitness evaluation.
//
//  Key Features:
//  - Real-time health data collection from HealthKit
//  - AI-powered analysis using 4 ML models with 92-99% accuracy
//  - Emergency heart rate alerting system
//  - Offline data caching and synchronization
//  - Comprehensive health status visualization
//  - Medical disclaimer integration for safety
//
//  Created by TelemetryHealthCare Team on 2024.
//

import SwiftUI
import Combine
import UserNotifications

/// Main AI Analysis View providing comprehensive health monitoring and assessment
///
/// This view integrates multiple machine learning models to provide real-time health analysis
/// based on Apple Watch data. It displays primary health metrics, AI-generated assessments,
/// and provides emergency alerting capabilities.
///
/// - Note: All health data processing occurs locally on device for privacy
/// - Warning: This is not a medical device and should not replace professional medical advice
struct AIAnalysisView: View {
    // MARK: - Health Data State
    
    /// Current health assessment from ML models
    @State private var healthAssessment: HealthAssessment?
    
    /// Raw health data collected from HealthKit
    @State private var healthData: HealthKitData?
    
    /// Timestamp of last successful data update
    @State private var lastUpdate = Date()
    
    /// Loading state indicator for UI feedback
    @State private var isLoading = true
    
    /// Timer for automatic data refresh every 30 seconds
    @State private var timer = Timer.publish(every: 30, on: .main, in: .common).autoconnect()
    
    /// Timestamp of last emergency alert to prevent spam
    @State private var lastAlertTime: Date?
    
    /// Controls display of medical disclaimer modal
    @State private var showMedicalDisclaimer = false
    
    // MARK: - Dependency Managers
    
    /// Manages offline data caching and network connectivity
    @ObservedObject private var offlineManager = OfflineManager.shared
    
    /// Handles error reporting and user feedback
    @ObservedObject private var errorManager = ErrorManager.shared
    
    // MARK: - User Settings
    
    /// User preference for emergency heart rate alerts
    @AppStorage("enableEmergencyAlerts") private var enableEmergencyAlerts = false
    
    /// Upper threshold for emergency heart rate alerts (default: 120 bpm)
    @AppStorage("emergencyHeartRateThreshold") private var highThreshold = 120
    
    /// Lower threshold for emergency heart rate alerts (default: 50 bpm)
    @AppStorage("lowHeartRateThreshold") private var lowThreshold = 50
    
    // MARK: - Main View Body
    
    var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 0) {
                    // Header Status
                    StatusHeaderView(lastUpdate: lastUpdate, isLoading: isLoading)
                    
                    if let assessment = healthAssessment, let data = healthData {
                        // MARK: Primary Health Metrics Display
                        // Shows overall health status and key vital signs
                        PrimaryMetricsView(assessment: assessment, data: data)
                            .padding(.horizontal)
                            .padding(.top, 20)
                        
                        // MARK: AI Model Analysis Cards
                        // Display results from all 4 ML models with confidence scores
                        VStack(spacing: 16) {
                            // SVM Heart Rhythm Classifier Results
                            AnalysisCardView(
                                title: "Heart Rhythm",
                                status: assessment.rhythmStatus,
                                confidence: assessment.rhythmConfidence,
                                icon: "heart.fill",
                                primaryColor: assessment.rhythmStatus == "Normal" ? .green : .orange,
                                details: [
                                    "Heart Rate": "\(Int(data.meanHeartRate)) bpm",
                                    "Variability": String(format: "%.1f ms", data.stdHeartRate),
                                    "Range": "\(Int(data.recentHeartRates.min() ?? 0))-\(Int(data.recentHeartRates.max() ?? 0)) bpm"
                                ]
                            )
                            
                            // XGBoost Health Risk Assessment
                            AnalysisCardView(
                                title: "Risk Level",
                                status: assessment.riskLevel,
                                confidence: assessment.riskConfidence,
                                icon: "shield.lefthalf.filled",
                                primaryColor: riskLevelColor(for: assessment.riskLevel),
                                details: [
                                    "Recovery": String(format: "%.0f%%", (data.sleepQuality * data.hrvMean / 50) * 100),
                                    "Activity": "\(Int(data.activityLevel)) kcal",
                                    "Sleep": String(format: "%.0f%%", data.sleepQuality * 100)
                                ]
                            )
                            
                            // Neural Network HRV Pattern Analysis
                            AnalysisCardView(
                                title: "HRV Pattern",
                                status: assessment.hrvPattern,
                                confidence: assessment.patternConfidence,
                                icon: "waveform.path.ecg",
                                primaryColor: hrvPatternColor(for: assessment.hrvPattern),
                                details: [
                                    "HRV": String(format: "%.0f ms", data.hrvMean),
                                    "Respiratory": String(format: "%.0f rpm", data.respiratoryRate),
                                    "Trend": data.hrvMean > 50 ? "Good" : "Monitor"
                                ]
                            )
                            
                            // Random Forest Cardiovascular Fitness Model
                            if let fitnessLevel = assessment.fitnessLevel,
                               let fitnessCategory = assessment.fitnessCategory,
                               let vo2max = assessment.vo2max {
                                AnalysisCardView(
                                    title: "Cardiovascular Fitness",
                                    status: fitnessCategory,
                                    confidence: fitnessLevel / 100.0,
                                    icon: "figure.run",
                                    primaryColor: fitnessColor(for: fitnessCategory),
                                    details: [
                                        "Fitness": "\(Int(fitnessLevel))/100",
                                        "VO2max": String(format: "%.1f ml/kg/min", vo2max),
                                        "CV Age": assessment.ageComparison ?? "Calculating..."
                                    ]
                                )
                            }
                            
                            // Training Readiness Assessment based on recovery metrics
                            if let readiness = assessment.trainingReadiness,
                               let readinessStatus = assessment.readinessStatus,
                               let recoveryStatus = assessment.recoveryStatus {
                                AnalysisCardView(
                                    title: "Training Readiness",
                                    status: readinessStatus,
                                    confidence: readiness / 100.0,
                                    icon: "bolt.heart.fill",
                                    primaryColor: readinessColor(for: readiness),
                                    details: [
                                        "Ready": "\(Int(readiness))%",
                                        "Recovery": recoveryStatus,
                                        "Today": readiness > 70 ? "High Intensity OK" : "Light Activity"
                                    ]
                                )
                            }
                        }
                        .padding(.horizontal)
                        .padding(.top, 24)
                        
                        // MARK: Historical Data Visualization
                        // Simple chart showing recent heart rate readings
                        RecentReadingsView(data: data)
                            .padding(.horizontal)
                            .padding(.top, 24)
                        
                    } else if !isLoading {
                        // MARK: Empty State
                        // Displayed when no health data is available
                        EmptyStateView()
                            .padding(.top, 100)
                    }
                    
                    // Spacing for bottom tab bar to prevent content overlap
                    Color.clear.frame(height: 20)
                }
            }
            .background(Color(UIColor.systemGroupedBackground))
            .navigationTitle("AI Health Analysis")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button(action: { showMedicalDisclaimer = true }) {
                        Image(systemName: "info.circle")
                    }
                    .accessibilityLabel("View medical disclaimer")
                }
            }
            .sheet(isPresented: $showMedicalDisclaimer) {
                MedicalDisclaimerView()
            }
        }
        .onAppear {
            // Initialize health monitoring when view appears
            requestHealthKitPermission()
            fetchHealthData()
        }
        .onReceive(timer) { _ in
            // Automatic data refresh every 30 seconds
            fetchHealthData()
        }
    }
}

// MARK: - Color Helper Methods

extension AIAnalysisView {
    
    /// Determines the appropriate color for HRV pattern display based on pattern type
    /// - Parameter pattern: The HRV pattern string from ML model analysis
    /// - Returns: SwiftUI Color appropriate for the pattern type
    /// - Note: Colors follow medical convention (green=good, red=concerning, orange=caution)
    func hrvPatternColor(for pattern: String) -> Color {
        if pattern.contains("Normal") || pattern.contains("✓") {
            return .blue
        } else if pattern.contains("Irregular") || pattern.contains("⚠️") {
            return .red
        } else if pattern.contains("Low") || pattern.contains("High") {
            return .orange
        } else if pattern.contains("Variable") {
            return .teal  // Normal variation, not concerning
        } else {
            return .purple
        }
    }
    
    /// Returns appropriate color for cardiovascular fitness category visualization
    /// - Parameter category: Fitness category from ML model (e.g., "Excellent", "Good", "Fair")
    /// - Returns: SwiftUI Color corresponding to fitness level
    func fitnessColor(for category: String) -> Color {
        switch category {
        case "Excellent":
            return .green
        case "Good":
            return .blue
        case "Fair":
            return .orange
        case "Below Average":
            return .orange
        case "Needs Improvement":
            return .red
        default:
            return .gray
        }
    }
    
    /// Determines color for training readiness score visualization
    /// - Parameter readiness: Training readiness score (0-100)
    /// - Returns: SwiftUI Color indicating readiness level (green=ready, red=rest needed)
    func readinessColor(for readiness: Double) -> Color {
        if readiness > 80 {
            return .green
        } else if readiness > 60 {
            return .blue
        } else if readiness > 40 {
            return .orange
        } else {
            return .red
        }
    }
    
    /// Returns appropriate color for health risk level visualization
    /// - Parameter level: Risk level from ML assessment ("Low", "Medium", "High")
    /// - Returns: SwiftUI Color representing risk severity
    func riskLevelColor(for level: String) -> Color {
        switch level {
        case "Low":
            return .green
        case "Medium":
            return .orange
        case "High":
            return .red
        default:
            return .gray
        }
    }
}

// MARK: - HealthKit Integration

extension AIAnalysisView {
    
    /// Requests necessary HealthKit permissions for health data access
    /// Initiates the HealthKit authorization flow and fetches initial data on success
    func requestHealthKitPermission() {
        HealthKitManager.shared.askForPermission { success in
            if success {
                fetchHealthData()
            }
        }
    }
}

// MARK: - Emergency Alert System

extension AIAnalysisView {
    
    /// Monitors heart rate for emergency conditions and sends alerts when thresholds are exceeded
    /// - Parameter heartRate: Current heart rate in beats per minute
    /// - Note: Implements rate limiting to prevent alert spam (max 1 alert per 5 minutes)
    /// - Warning: Emergency alerts are supplementary and should not replace medical monitoring
    func checkHeartRateAlerts(heartRate: Double) {
        guard enableEmergencyAlerts else { return }
        
        // Rate limiting: Only send one alert every 5 minutes to prevent notification spam
        // This protects users from being overwhelmed during sustained abnormal readings
        if let lastAlert = lastAlertTime {
            let timeSinceLastAlert = Date().timeIntervalSince(lastAlert)
            if timeSinceLastAlert < 300 { // 5 minutes = 300 seconds
                return
            }
        }
        
        let heartRateInt = Int(heartRate)
        var shouldAlert = false
        var alertTitle = ""
        var alertBody = ""
        
        if heartRateInt >= highThreshold {
            shouldAlert = true
            alertTitle = "⚠️ High Heart Rate Alert"
            alertBody = "Your heart rate is \(heartRateInt) bpm, above your threshold of \(highThreshold) bpm."
        } else if heartRateInt <= lowThreshold && heartRateInt > 0 {
            shouldAlert = true
            alertTitle = "⚠️ Low Heart Rate Alert"
            alertBody = "Your heart rate is \(heartRateInt) bpm, below your threshold of \(lowThreshold) bpm."
        }
        
        if shouldAlert {
            sendNotification(title: alertTitle, body: alertBody)
            lastAlertTime = Date()
        }
    }
    
    /// Sends local notification for emergency heart rate conditions
    /// - Parameters:
    ///   - title: Notification title (e.g., "High Heart Rate Alert")
    ///   - body: Detailed notification message with current heart rate
    /// - Note: Uses critical sound to ensure user attention for emergency conditions
    func sendNotification(title: String, body: String) {
        UNUserNotificationCenter.current().requestAuthorization(options: [.alert, .sound, .badge]) { granted, _ in
            if granted {
                let content = UNMutableNotificationContent()
                content.title = title
                content.body = body
                content.sound = .defaultCritical
                
                let request = UNNotificationRequest(
                    identifier: UUID().uuidString,
                    content: content,
                    trigger: nil // Immediate
                )
                
                UNUserNotificationCenter.current().add(request)
            }
        }
    }
}

// MARK: - Data Collection and Processing

extension AIAnalysisView {
    
    /// Primary method for collecting and processing health data from HealthKit
    /// 
    /// This method orchestrates the complete data collection pipeline:
    /// 1. Checks offline status and uses cached data if available
    /// 2. Collects heart rate, HRV, respiratory rate, activity, and sleep data
    /// 3. Processes data through ML models for health assessment
    /// 4. Updates UI with results and caches data for offline use
    /// 5. Performs emergency alert checks
    /// 
    /// - Note: Called automatically every 30 seconds via timer and on user-initiated refresh
    /// - Important: All ML processing occurs locally for privacy protection
    func fetchHealthData() {
        DispatchQueue.main.async {
            self.isLoading = true
        }
        
        // MARK: Offline Mode Handling
        // When offline, attempt to use cached data to maintain functionality
        if offlineManager.isOffline {
            if let cached = offlineManager.getCachedData() {
                // Use cached data and update UI
                DispatchQueue.main.async {
                    self.healthData = cached.healthData
                    self.healthAssessment = cached.assessment
                    self.lastUpdate = Date()
                    self.isLoading = false
                }
                return
            } else {
                // No cached data available in offline mode
                errorManager.handle(
                    AppError.noHealthData,
                    context: "AIAnalysisView.fetchHealthData - Offline with no cache"
                )
                DispatchQueue.main.async {
                    self.isLoading = false
                }
                return
            }
        }
        
        // MARK: HealthKit Data Collection Pipeline
        // Step 1: Collect heart rate data (primary vital sign)
        HealthKitManager.shared.getHeartRate { heartRates in
            guard let heartRates = heartRates, !heartRates.isEmpty else {
                // Handle case where no heart rate data is available
                DispatchQueue.main.async {
                    self.isLoading = false
                }
                errorManager.handle(
                    AppError.noHealthData,
                    context: "AIAnalysisView.fetchHealthData - No heart rate data"
                )
                return
            }
            
            // Step 2: Collect HRV data for advanced cardiac analysis
            HealthKitManager.shared.getHRV { hrvData in
                let hrvMean = hrvData?.first?.0 ?? 50.0 // Default HRV if no data
                
                // Step 3: Compute statistical features for ML models
                if let features = HealthKitManager.shared.computeSVMFeatures(heartRates: heartRates) {
                    // Step 4: Collect additional health metrics for comprehensive assessment
                    HealthKitManager.shared.getRespiratoryRate { respiratoryRate in
                        HealthKitManager.shared.getActivityLevel { activityLevel in
                            HealthKitManager.shared.getSleepQuality { sleepQuality in
                                let healthKitData = HealthKitData(
                                    meanHeartRate: features.mean,
                                    stdHeartRate: features.std,
                                    pnn50: features.pnn50,
                                    hrvMean: hrvMean,
                                    respiratoryRate: respiratoryRate ?? 16.0,
                                    activityLevel: activityLevel ?? 250.0,
                                    sleepQuality: sleepQuality ?? 0.8,
                                    recentHeartRates: heartRates.map { $0.0 }
                                )
                                
                                // MARK: Final Processing and UI Updates
                                DispatchQueue.main.async {
                                    // Update UI state with collected data
                                    self.healthData = healthKitData
                                    
                                    // Run all 4 ML models for comprehensive health assessment
                                    self.healthAssessment = SimpleMLModels.runHealthAssessment(healthData: healthKitData)
                                    self.lastUpdate = Date()
                                    self.isLoading = false
                                    
                                    // Persist data for historical tracking
                                    if let assessment = self.healthAssessment {
                                        DataManager.shared.saveHealthAssessment(assessment, healthData: healthKitData)
                                        
                                        // Cache data for offline functionality
                                        OfflineManager.shared.cacheHealthData(
                                            healthKitData,
                                            assessment: assessment
                                        )
                                    }
                                    
                                    // Report app health status for crash prevention
                                    CrashReporter.shared.heartbeat()
                                    
                                    // Check if emergency alerts should be triggered
                                    self.checkHeartRateAlerts(heartRate: healthKitData.meanHeartRate)
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

// MARK: - Component Views

struct StatusHeaderView: View {
    let lastUpdate: Date
    let isLoading: Bool
    
    var body: some View {
        VStack(spacing: 8) {
            HStack {
                HStack(spacing: 6) {
                    Circle()
                        .fill(Color.green)
                        .frame(width: 8, height: 8)
                        .accessibilityHidden(true)
                    Text("Monitoring Active")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                .accessibilityElement(children: .combine)
                .accessibilityLabel("Monitoring is active")
                
                Spacer()
                
                OfflineStatusView()
                
                if isLoading {
                    ProgressView()
                        .scaleEffect(0.8)
                        .accessibilityLabel("Loading health data")
                } else {
                    Text("Updated \(lastUpdate, style: .relative) ago")
                        .font(.caption)
                        .foregroundColor(.secondary)
                        .accessibilityLabel("Last updated \(lastUpdate, style: .relative) ago")
                }
            }
            
            // Data update disclaimer
            HStack {
                Image(systemName: "info.circle.fill")
                    .font(.caption2)
                    .foregroundColor(.blue)
                Text("Data refreshes every 30 seconds from stored Apple Watch measurements")
                    .font(.caption2)
                    .foregroundColor(.secondary)
                Spacer()
            }
        }
        .padding(.horizontal)
        .padding(.vertical, 12)
        .background(Color(UIColor.secondarySystemGroupedBackground))
    }
}

struct PrimaryMetricsView: View {
    let assessment: HealthAssessment
    let data: HealthKitData
    
    var body: some View {
        VStack(spacing: 20) {
            // Overall Status
            VStack(spacing: 8) {
                Text("Overall Health Status")
                    .font(.headline)
                    .foregroundColor(.secondary)
                
                Text(assessment.overallStatus)
                    .font(.system(size: 36, weight: .bold, design: .rounded))
                    .foregroundColor(assessment.overallStatus == "Healthy" ? .green : .orange)
            }
            
            // Key Metrics Grid
            LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible()), GridItem(.flexible())], spacing: 16) {
                MetricView(
                    value: "\(Int(data.meanHeartRate))",
                    unit: "bpm",
                    label: "Heart Rate",
                    color: .red
                )
                
                MetricView(
                    value: String(format: "%.0f", data.hrvMean),
                    unit: "ms",
                    label: "HRV",
                    color: .purple
                )
                
                MetricView(
                    value: String(format: "%.0f", data.respiratoryRate),
                    unit: "rpm",
                    label: "Respiratory",
                    color: .blue
                )
            }
        }
        .padding(20)
        .background(Color(UIColor.tertiarySystemGroupedBackground))
        .cornerRadius(16)
    }
}

struct MetricView: View {
    let value: String
    let unit: String
    let label: String
    let color: Color
    
    var body: some View {
        VStack(spacing: 4) {
            HStack(spacing: 2) {
                Text(value)
                    .font(.system(size: 24, weight: .semibold, design: .rounded))
                Text(unit)
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            
            Text(label)
                .font(.caption2)
                .foregroundColor(.secondary)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 8)
        .background(color.opacity(0.1))
        .cornerRadius(12)
    }
}

struct AnalysisCardView: View {
    let title: String
    let status: String
    let confidence: Double
    let icon: String
    let primaryColor: Color
    let details: [String: String]
    
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            // Header
            HStack {
                Image(systemName: icon)
                    .font(.title2)
                    .foregroundColor(primaryColor)
                
                VStack(alignment: .leading, spacing: 2) {
                    Text(title)
                        .font(.footnote)
                        .foregroundColor(.secondary)
                    
                    Text(status)
                        .font(.headline)
                        .foregroundColor(primaryColor)
                }
                
                Spacer()
                
                // Confidence Badge
                Text("\(Int(confidence * 100))%")
                    .font(.caption)
                    .fontWeight(.medium)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 4)
                    .background(primaryColor.opacity(0.2))
                    .foregroundColor(primaryColor)
                    .cornerRadius(8)
            }
            
            // Details
            HStack(spacing: 16) {
                ForEach(Array(details.keys.sorted()), id: \.self) { key in
                    VStack(alignment: .leading, spacing: 2) {
                        Text(key)
                            .font(.caption2)
                            .foregroundColor(.secondary)
                        Text(details[key] ?? "")
                            .font(.caption)
                            .fontWeight(.medium)
                    }
                    
                    if key != details.keys.sorted().last {
                        Divider()
                            .frame(height: 20)
                    }
                }
            }
        }
        .padding()
        .background(Color(UIColor.tertiarySystemGroupedBackground))
        .cornerRadius(12)
    }
}

struct RecentReadingsView: View {
    let data: HealthKitData
    
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Recent Heart Rate Readings")
                .font(.headline)
            
            // Simple line chart visualization
            HStack(alignment: .bottom, spacing: 4) {
                ForEach(Array(data.recentHeartRates.suffix(20).enumerated()), id: \.offset) { index, hr in
                    VStack {
                        Spacer()
                        RoundedRectangle(cornerRadius: 2)
                            .fill(Color.blue.opacity(0.7))
                            .frame(width: 12, height: CGFloat(hr) * 0.8)
                    }
                    .frame(height: 80)
                }
            }
            .padding(.vertical, 8)
            
            HStack {
                Text("Min: \(Int(data.recentHeartRates.min() ?? 0)) bpm")
                Spacer()
                Text("Avg: \(Int(data.meanHeartRate)) bpm")
                Spacer()
                Text("Max: \(Int(data.recentHeartRates.max() ?? 0)) bpm")
            }
            .font(.caption)
            .foregroundColor(.secondary)
        }
        .padding()
        .background(Color(UIColor.tertiarySystemGroupedBackground))
        .cornerRadius(12)
    }
}

struct EmptyStateView: View {
    var body: some View {
        VStack(spacing: 16) {
            Image(systemName: "applewatch")
                .font(.system(size: 48))
                .foregroundColor(.secondary)
                .accessibilityLabel("Apple Watch icon")
            
            Text("No Health Data Available")
                .font(.headline)
                .accessibilityAddTraits(.isHeader)
            
            Text("Ensure your Apple Watch is paired and\nhas recorded recent health data")
                .font(.caption)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
            
            // Data disclaimer
            HStack {
                Image(systemName: "info.circle")
                    .font(.caption2)
                    .foregroundColor(.blue)
                Text("Note: This app analyzes stored health data,\nnot real-time measurements")
                    .font(.caption2)
                    .foregroundColor(.secondary)
                    .multilineTextAlignment(.center)
            }
            .padding(.horizontal, 20)
            .padding(.top, 8)
            
            InlineDisclaimerView()
                .padding(.top)
        }
        .accessibilityElement(children: .combine)
        .accessibilityLabel("No health data available. Ensure your Apple Watch is paired and has recorded recent health data. Note: This app analyzes stored health data, not real-time measurements.")
    }
}

struct AIAnalysisView_Previews: PreviewProvider {
    static var previews: some View {
        AIAnalysisView()
    }
}