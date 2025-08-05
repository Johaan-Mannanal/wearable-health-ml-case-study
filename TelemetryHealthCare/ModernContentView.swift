import SwiftUI

// MARK: - Color Extensions
extension Color {
    static let healthPrimary = Color(red: 0.0, green: 0.48, blue: 1.0)
    static let healthSecondary = Color(red: 0.2, green: 0.78, blue: 0.35)
    static let healthWarning = Color(red: 1.0, green: 0.58, blue: 0.0)
    static let healthDanger = Color(red: 1.0, green: 0.23, blue: 0.19)
    
    static let cardBackground = Color(.systemBackground)
    static let cardBorder = Color(.separator).opacity(0.3)
}

struct ModernContentView: View {
    @State private var healthAssessment: HealthAssessment?
    @State private var isMonitoring = false
    @State private var lastUpdate = Date()
    @State private var errorMessage: String?
    @State private var healthData: HealthKitData?
    @State private var showDebugInfo = false
    
    let timer = Timer.publish(every: 5, on: .main, in: .common).autoconnect()
    
    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 24) {
                    // Status Header
                    StatusHeaderView(
                        isMonitoring: isMonitoring,
                        lastUpdate: lastUpdate,
                        showDebugInfo: $showDebugInfo
                    )
                    .padding(.horizontal)
                    
                    // Main Content
                    if let assessment = healthAssessment {
                        // Overall Health Status
                        OverallHealthCard(assessment: assessment)
                            .padding(.horizontal)
                        
                        // Analysis Cards
                        VStack(spacing: 16) {
                            ModernAnalysisCard(
                                title: "Heart Rhythm",
                                result: assessment.rhythmStatus,
                                confidence: assessment.rhythmConfidence,
                                icon: "heart.fill",
                                color: assessment.rhythmStatus == "Normal" ? .healthPrimary : .healthDanger,
                                metrics: healthData.map { [
                                    MetricItem(label: "Heart Rate", value: "\(Int($0.meanHeartRate)) bpm"),
                                    MetricItem(label: "Variability", value: String(format: "%.1f", $0.stdHeartRate)),
                                    MetricItem(label: "pNN50", value: String(format: "%.2f", $0.pnn50))
                                ]} ?? []
                            )
                            
                            ModernAnalysisCard(
                                title: "Health Risk",
                                result: assessment.riskLevel,
                                confidence: assessment.riskConfidence,
                                icon: "shield.fill",
                                color: assessment.riskLevel == "Low" ? .healthSecondary : .healthWarning,
                                metrics: healthData.map { [
                                    MetricItem(label: "HRV", value: String(format: "%.0f ms", $0.hrvMean)),
                                    MetricItem(label: "Activity", value: "\(Int($0.activityLevel)) cal"),
                                    MetricItem(label: "Sleep", value: String(format: "%.1f hrs", $0.sleepQuality * 8))
                                ]} ?? []
                            )
                            
                            ModernAnalysisCard(
                                title: "HRV Pattern",
                                result: assessment.hrvPattern,
                                confidence: assessment.patternConfidence,
                                icon: "waveform.path.ecg",
                                color: hrvPatternColor(for: assessment.hrvPattern),
                                metrics: []
                            )
                        }
                        .padding(.horizontal)
                    } else {
                        EmptyStateView(errorMessage: errorMessage)
                            .padding(.horizontal)
                    }
                    
                    // Debug Info (if enabled)
                    if showDebugInfo {
                        DebugInfoView()
                            .padding(.horizontal)
                    }
                }
                .padding(.vertical)
            }
            .navigationTitle("Rhythm 360")
            .navigationBarTitleDisplayMode(.large)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button(action: toggleMonitoring) {
                        HStack(spacing: 6) {
                            Image(systemName: isMonitoring ? "stop.circle.fill" : "play.circle.fill")
                                .font(.title3)
                            Text(isMonitoring ? "Stop" : "Start")
                                .font(.callout)
                                .fontWeight(.medium)
                        }
                        .foregroundStyle(isMonitoring ? Color.healthDanger : Color.healthPrimary)
                        .padding(.horizontal, 12)
                        .padding(.vertical, 6)
                        .background(
                            RoundedRectangle(cornerRadius: 20)
                                .fill(isMonitoring ? Color.healthDanger.opacity(0.15) : Color.healthPrimary.opacity(0.15))
                        )
                    }
                }
            }
        }
        .onAppear {
            requestHealthKitPermission()
        }
        .onReceive(timer) { _ in
            if isMonitoring {
                fetchHealthData()
            }
        }
    }
    
    // MARK: - Helper Functions
    func toggleMonitoring() {
        withAnimation(.spring(response: 0.6, dampingFraction: 0.8)) {
            isMonitoring.toggle()
        }
        if isMonitoring {
            fetchHealthData()
        }
    }
    
    func requestHealthKitPermission() {
        HealthKitManager.shared.askForPermission { success in
            DispatchQueue.main.async {
                if !success {
                    errorMessage = "HealthKit permission denied. Please enable in Settings > Privacy > Health."
                }
            }
        }
    }
    
    func fetchHealthData() {
        print("🔄 Starting health data fetch...")
        
        HealthKitManager.shared.getHeartRate { heartRates in
            guard let heartRates = heartRates, !heartRates.isEmpty else {
                DispatchQueue.main.async {
                    self.errorMessage = """
                    No heart rate data found. Please ensure:
                    • Apple Watch is paired and worn
                    • Health app has recorded data
                    • Permissions are granted
                    """
                    print("❌ Heart rate fetch failed - no data available")
                }
                return
            }
            
            print("✅ Received \(heartRates.count) heart rate samples")
            
            HealthKitManager.shared.getHRV { hrvData in
                let hrvMean = hrvData?.first?.0 ?? 50.0
                print("✅ HRV data: \(hrvData?.count ?? 0) samples")
                
                // Calculate features
                if let features = HealthKitManager.shared.computeSVMFeatures(heartRates: heartRates) {
                    let healthKitData = HealthKitData(
                        meanHeartRate: features.mean,
                        stdHeartRate: features.std,
                        pnn50: features.pnn50,
                        hrvMean: hrvMean,
                        respiratoryRate: 16.0, // Default value - need to implement
                        activityLevel: 250.0,  // Default value - need to implement
                        sleepQuality: 0.8,     // Default value - need to implement
                        recentHeartRates: heartRates.map { $0.0 }
                    )
                    
                    DispatchQueue.main.async {
                        withAnimation(.easeInOut(duration: 0.3)) {
                            self.healthData = healthKitData
                            self.healthAssessment = SimpleMLModels.runHealthAssessment(healthData: healthKitData)
                            self.lastUpdate = Date()
                            self.errorMessage = nil
                        }
                    }
                }
            }
        }
    }
    
    func hrvPatternColor(for pattern: String) -> Color {
        switch pattern {
        case "Normal": return .healthPrimary
        case "AFib": return .healthDanger
        case "Bradycardia": return .healthWarning
        case "Tachycardia": return .orange
        default: return .gray
        }
    }
}

// MARK: - Component Views
struct StatusHeaderView: View {
    let isMonitoring: Bool
    let lastUpdate: Date
    @Binding var showDebugInfo: Bool
    
    var body: some View {
        HStack {
            // Status Indicator
            HStack(spacing: 8) {
                Circle()
                    .fill(isMonitoring ? Color.healthSecondary : Color.gray)
                    .frame(width: 12, height: 12)
                    .overlay(
                        Circle()
                            .stroke(isMonitoring ? Color.healthSecondary.opacity(0.3) : Color.clear, lineWidth: 8)
                            .scaleEffect(isMonitoring ? 1.5 : 1)
                            .opacity(isMonitoring ? 0 : 1)
                            .animation(isMonitoring ? .easeOut(duration: 1.5).repeatForever(autoreverses: false) : .default, value: isMonitoring)
                    )
                
                VStack(alignment: .leading, spacing: 2) {
                    Text(isMonitoring ? "Monitoring Active" : "Not Monitoring")
                        .font(.subheadline)
                        .fontWeight(.medium)
                    
                    if isMonitoring {
                        Text("Updated \(lastUpdate, style: .relative) ago")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }
            }
            
            Spacer()
            
            // Debug Toggle
            Button(action: { showDebugInfo.toggle() }) {
                Image(systemName: "ant.circle")
                    .foregroundStyle(.secondary)
            }
        }
        .padding()
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 16))
    }
}

struct OverallHealthCard: View {
    let assessment: HealthAssessment
    
    var body: some View {
        HStack(spacing: 16) {
            Image(systemName: assessment.overallStatus == "Healthy" ? "checkmark.seal.fill" : "exclamationmark.shield.fill")
                .font(.system(size: 48))
                .foregroundStyle(assessment.overallStatus == "Healthy" ? Color.healthSecondary.gradient : Color.healthWarning.gradient)
            
            VStack(alignment: .leading, spacing: 4) {
                Text("Overall Health Status")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                
                Text(assessment.overallStatus)
                    .font(.title2)
                    .fontWeight(.bold)
                
                Text("Based on \(Date(), style: .date)")
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
            }
            
            Spacer()
        }
        .padding()
        .frame(maxWidth: .infinity)
        .background(
            RoundedRectangle(cornerRadius: 20)
                .fill(.regularMaterial)
                .overlay(
                    RoundedRectangle(cornerRadius: 20)
                        .stroke(Color.cardBorder, lineWidth: 1)
                )
        )
    }
}

struct MetricItem {
    let label: String
    let value: String
}

struct ModernAnalysisCard: View {
    let title: String
    let result: String
    let confidence: Double
    let icon: String
    let color: Color
    let metrics: [MetricItem]
    
    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            // Header
            HStack {
                Image(systemName: icon)
                    .font(.title2)
                    .foregroundStyle(color.gradient)
                
                Text(title)
                    .font(.headline)
                
                Spacer()
                
                // Confidence Badge
                Text("\(Int(confidence * 100))%")
                    .font(.caption)
                    .fontWeight(.medium)
                    .padding(.horizontal, 10)
                    .padding(.vertical, 4)
                    .background(color.opacity(0.2), in: Capsule())
                    .foregroundStyle(color)
            }
            
            // Result
            Text(result)
                .font(.title3)
                .fontWeight(.semibold)
                .foregroundStyle(color)
            
            // Metrics
            if !metrics.isEmpty {
                Divider()
                
                HStack(spacing: 20) {
                    ForEach(metrics.indices, id: \.self) { index in
                        VStack(alignment: .leading, spacing: 4) {
                            Text(metrics[index].label)
                                .font(.caption2)
                                .foregroundStyle(.secondary)
                            
                            Text(metrics[index].value)
                                .font(.caption)
                                .fontWeight(.medium)
                        }
                        
                        if index < metrics.count - 1 {
                            Divider()
                                .frame(height: 30)
                        }
                    }
                }
            }
        }
        .padding()
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(
            RoundedRectangle(cornerRadius: 16)
                .fill(.regularMaterial)
                .overlay(
                    RoundedRectangle(cornerRadius: 16)
                        .stroke(Color.cardBorder, lineWidth: 1)
                )
        )
    }
}

struct EmptyStateView: View {
    let errorMessage: String?
    
    var body: some View {
        VStack(spacing: 24) {
            Image(systemName: "applewatch.watchface")
                .font(.system(size: 80))
                .foregroundStyle(.tertiary)
            
            VStack(spacing: 8) {
                Text("No Health Data")
                    .font(.title2)
                    .fontWeight(.semibold)
                
                Text(errorMessage ?? "Start monitoring to see your health insights")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
            }
        }
        .padding(40)
        .frame(maxWidth: .infinity)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 20))
    }
}

struct DebugInfoView: View {
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Debug Information")
                .font(.caption)
                .fontWeight(.semibold)
                .foregroundStyle(.secondary)
            
            Text("View console logs in Xcode for detailed information")
                .font(.caption2)
                .foregroundStyle(.tertiary)
        }
        .padding()
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 12))
    }
}

struct ModernContentView_Previews: PreviewProvider {
    static var previews: some View {
        ModernContentView()
    }
}