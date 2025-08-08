//
//  AIAnalysisView.swift
//  Rhythm 360
//
//  Main AI health analysis view with real-time monitoring
//

import SwiftUI
import Combine
import UserNotifications

struct AIAnalysisView: View {
    @State private var healthAssessment: HealthAssessment?
    @State private var healthData: HealthKitData?
    @State private var lastUpdate = Date()
    @State private var isLoading = true
    @State private var timer = Timer.publish(every: 30, on: .main, in: .common).autoconnect()
    @State private var lastAlertTime: Date?
    
    // Settings
    @AppStorage("enableEmergencyAlerts") private var enableEmergencyAlerts = false
    @AppStorage("emergencyHeartRateThreshold") private var highThreshold = 120
    @AppStorage("lowHeartRateThreshold") private var lowThreshold = 50
    
    var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 0) {
                    // Header Status
                    StatusHeaderView(lastUpdate: lastUpdate, isLoading: isLoading)
                    
                    if let assessment = healthAssessment, let data = healthData {
                        // Primary Metrics
                        PrimaryMetricsView(assessment: assessment, data: data)
                            .padding(.horizontal)
                            .padding(.top, 20)
                        
                        // AI Analysis Cards
                        VStack(spacing: 16) {
                            // Heart Rhythm Analysis
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
                            
                            // Risk Assessment
                            AnalysisCardView(
                                title: "Risk Level",
                                status: assessment.riskLevel,
                                confidence: assessment.riskConfidence,
                                icon: "shield.lefthalf.filled",
                                primaryColor: assessment.riskLevel == "Low" ? .green : .red,
                                details: [
                                    "Recovery": String(format: "%.0f%%", (data.sleepQuality * data.hrvMean / 50) * 100),
                                    "Activity": "\(Int(data.activityLevel)) kcal",
                                    "Sleep": String(format: "%.0f%%", data.sleepQuality * 100)
                                ]
                            )
                            
                            // HRV Pattern
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
                        }
                        .padding(.horizontal)
                        .padding(.top, 24)
                        
                        // Recent Readings
                        RecentReadingsView(data: data)
                            .padding(.horizontal)
                            .padding(.top, 24)
                        
                    } else if !isLoading {
                        // Empty State
                        EmptyStateView()
                            .padding(.top, 100)
                    }
                    
                    // Bottom padding for tab bar
                    Color.clear.frame(height: 20)
                }
            }
            .background(Color(UIColor.systemGroupedBackground))
            .navigationTitle("")
            .navigationBarTitleDisplayMode(.inline)
        }
        .onAppear {
            requestHealthKitPermission()
            fetchHealthData()
        }
        .onReceive(timer) { _ in
            fetchHealthData()
        }
    }
    
    func hrvPatternColor(for pattern: String) -> Color {
        if pattern.contains("Normal") || pattern.contains("✓") {
            return .blue
        } else if pattern.contains("Irregular") || pattern.contains("⚠️") {
            return .red
        } else if pattern.contains("Low") {
            return .orange
        } else if pattern.contains("High") {
            return .orange
        } else {
            return .purple
        }
    }
    
    func requestHealthKitPermission() {
        HealthKitManager.shared.askForPermission { success in
            if success {
                fetchHealthData()
            }
        }
    }
    
    func checkHeartRateAlerts(heartRate: Double) {
        guard enableEmergencyAlerts else { return }
        
        // Only send one alert every 5 minutes to avoid spam
        if let lastAlert = lastAlertTime {
            let timeSinceLastAlert = Date().timeIntervalSince(lastAlert)
            if timeSinceLastAlert < 300 { // 5 minutes
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
    
    func fetchHealthData() {
        DispatchQueue.main.async {
            self.isLoading = true
        }
        
        HealthKitManager.shared.getHeartRate { heartRates in
            guard let heartRates = heartRates, !heartRates.isEmpty else {
                DispatchQueue.main.async {
                    self.isLoading = false
                }
                return
            }
            
            HealthKitManager.shared.getHRV { hrvData in
                let hrvMean = hrvData?.first?.0 ?? 50.0
                
                if let features = HealthKitManager.shared.computeSVMFeatures(heartRates: heartRates) {
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
                                
                                DispatchQueue.main.async {
                                    self.healthData = healthKitData
                                    self.healthAssessment = SimpleMLModels.runHealthAssessment(healthData: healthKitData)
                                    self.lastUpdate = Date()
                                    self.isLoading = false
                                    
                                    // Save to Core Data
                                    DataManager.shared.saveHealthAssessment(self.healthAssessment!, healthData: healthKitData)
                                    
                                    // Check for alerts
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
        HStack {
            HStack(spacing: 6) {
                Circle()
                    .fill(Color.green)
                    .frame(width: 8, height: 8)
                Text("Monitoring Active")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            
            Spacer()
            
            if isLoading {
                ProgressView()
                    .scaleEffect(0.8)
            } else {
                Text("Updated \(lastUpdate, style: .relative) ago")
                    .font(.caption)
                    .foregroundColor(.secondary)
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
            
            Text("No Health Data Available")
                .font(.headline)
            
            Text("Ensure your Apple Watch is paired and\nhas recorded recent health data")
                .font(.caption)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
        }
    }
}

struct AIAnalysisView_Previews: PreviewProvider {
    static var previews: some View {
        AIAnalysisView()
    }
}