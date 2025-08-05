import SwiftUI

struct ContentView: View {
    @State private var healthAssessment: HealthAssessment?
    @State private var isMonitoring = false
    @State private var lastUpdate = Date()
    @State private var errorMessage: String?
    @State private var healthData: HealthKitData?
    
    let timer = Timer.publish(every: 5, on: .main, in: .common).autoconnect()
    
    var body: some View {
        VStack(spacing: 20) {
            // Header
            VStack(spacing: 8) {
                Text("Rhythm 360")
                    .font(.largeTitle)
                    .fontWeight(.bold)
                
                Text("AI-Powered Health Monitoring")
                    .font(.subheadline)
                    .foregroundColor(.secondary)
            }
            .padding(.top)
            
            // Status
            HStack {
                Circle()
                    .fill(isMonitoring ? Color.green : Color.gray)
                    .frame(width: 10, height: 10)
                
                Text(isMonitoring ? "Monitoring Active" : "Not Monitoring")
                    .font(.caption)
                    .foregroundColor(.secondary)
                
                Spacer()
                
                if isMonitoring {
                    Text("Updated: \(lastUpdate, style: .time)")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
            }
            .padding(.horizontal)
            
            // Health Metrics Display
            if let assessment = healthAssessment {
                ScrollView {
                    VStack(spacing: 16) {
                        // Overall Status
                        HealthStatusCard(
                            title: "Overall Health Status",
                            status: assessment.overallStatus,
                            icon: assessment.overallStatus == "Healthy" ? "checkmark.circle.fill" : "exclamationmark.triangle.fill",
                            color: assessment.overallStatus == "Healthy" ? .green : .orange
                        )
                        
                        // Heart Rhythm Analysis
                        AnalysisCard(
                            title: "Heart Rhythm",
                            result: assessment.rhythmStatus,
                            confidence: assessment.rhythmConfidence,
                            icon: "heart.fill",
                            color: assessment.rhythmStatus == "Normal" ? .blue : .red,
                            metrics: healthData.map { [
                                "Mean HR": "\(Int($0.meanHeartRate)) bpm",
                                "Variability": String(format: "%.1f", $0.stdHeartRate),
                                "pNN50": String(format: "%.2f", $0.pnn50)
                            ]} ?? [:]
                        )
                        
                        // Health Risk Assessment
                        AnalysisCard(
                            title: "Health Risk",
                            result: assessment.riskLevel,
                            confidence: assessment.riskConfidence,
                            icon: "shield.fill",
                            color: assessment.riskLevel == "Low" ? .green : .orange,
                            metrics: healthData.map { [
                                "HRV": String(format: "%.1f ms", $0.hrvMean),
                                "Activity": "\(Int($0.activityLevel)) cal",
                                "Sleep": String(format: "%.1f hrs", $0.sleepQuality * 8)
                            ]} ?? [:]
                        )
                        
                        // HRV Pattern Analysis
                        AnalysisCard(
                            title: "HRV Pattern",
                            result: assessment.hrvPattern,
                            confidence: assessment.patternConfidence,
                            icon: "waveform.path.ecg",
                            color: assessment.hrvPattern == "Normal" ? .purple : .red,
                            metrics: [:]
                        )
                    }
                    .padding()
                }
            } else {
                // No Data State
                VStack(spacing: 20) {
                    Image(systemName: "heart.text.square")
                        .font(.system(size: 60))
                        .foregroundColor(.gray)
                    
                    Text("No health data available")
                        .font(.headline)
                        .foregroundColor(.secondary)
                    
                    Text("Connect your Apple Watch and start monitoring")
                        .font(.caption)
                        .foregroundColor(.secondary)
                        .multilineTextAlignment(.center)
                }
                .padding()
                
                Spacer()
            }
            
            // Error Message
            if let error = errorMessage {
                Text(error)
                    .font(.caption)
                    .foregroundColor(.red)
                    .padding(.horizontal)
            }
            
            // Action Button
            Button(action: toggleMonitoring) {
                HStack {
                    Image(systemName: isMonitoring ? "stop.circle" : "play.circle")
                    Text(isMonitoring ? "Stop Monitoring" : "Start Monitoring")
                }
                .font(.headline)
                .padding()
                .frame(maxWidth: .infinity)
                .background(isMonitoring ? Color.red : Color.blue)
                .foregroundColor(.white)
                .cornerRadius(10)
            }
            .padding(.horizontal)
            .padding(.bottom)
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
    
    func toggleMonitoring() {
        isMonitoring.toggle()
        if isMonitoring {
            fetchHealthData()
        }
    }
    
    func requestHealthKitPermission() {
        HealthKitManager.shared.askForPermission { success in
            DispatchQueue.main.async {
                if !success {
                    errorMessage = "HealthKit permission denied. Please enable in Settings."
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

// MARK: - Component Views
struct HealthStatusCard: View {
    let title: String
    let status: String
    let icon: String
    let color: Color
    
    var body: some View {
        HStack {
            Image(systemName: icon)
                .font(.largeTitle)
                .foregroundColor(color)
            
            VStack(alignment: .leading) {
                Text(title)
                    .font(.caption)
                    .foregroundColor(.secondary)
                
                Text(status)
                    .font(.title2)
                    .fontWeight(.semibold)
            }
            
            Spacer()
        }
        .padding()
        .background(color.opacity(0.1))
        .cornerRadius(12)
    }
}

struct AnalysisCard: View {
    let title: String
    let result: String
    let confidence: Double
    let icon: String
    let color: Color
    let metrics: [String: String]
    
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Image(systemName: icon)
                    .foregroundColor(color)
                
                Text(title)
                    .font(.headline)
                
                Spacer()
                
                Text("\(Int(confidence * 100))% confident")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            
            Text(result)
                .font(.title3)
                .fontWeight(.semibold)
                .foregroundColor(color)
            
            if !metrics.isEmpty {
                Divider()
                
                HStack {
                    ForEach(Array(metrics.keys.sorted()), id: \.self) { key in
                        VStack(alignment: .leading) {
                            Text(key)
                                .font(.caption2)
                                .foregroundColor(.secondary)
                            
                            Text(metrics[key] ?? "")
                                .font(.caption)
                                .fontWeight(.medium)
                        }
                        
                        if key != metrics.keys.sorted().last {
                            Spacer()
                        }
                    }
                }
            }
        }
        .padding()
        .background(Color.gray.opacity(0.1))
        .cornerRadius(12)
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
