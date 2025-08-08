//
//  SettingsView.swift
//  Rhythm 360
//
//  Settings and preferences page
//

import SwiftUI

struct SettingsView: View {
    @AppStorage("enableNotifications") private var enableNotifications = true
    @AppStorage("enableEmergencyAlerts") private var enableEmergencyAlerts = false
    @AppStorage("emergencyHeartRateThreshold") private var emergencyHeartRateThreshold = 120
    @AppStorage("lowHeartRateThreshold") private var lowHeartRateThreshold = 50
    @StateObject private var dataManager = DataManager.shared
    @State private var showingDeleteAlert = false
    @State private var showingAbout = false
    @State private var showingExportAlert = false
    @State private var exportAlertMessage = ""
    
    var body: some View {
        NavigationView {
            List {
                // Health Monitoring Section
                Section {
                    // Notifications
                    Toggle(isOn: $enableNotifications) {
                        Label {
                            VStack(alignment: .leading, spacing: 2) {
                                Text("Health Notifications")
                                Text("Get alerts for significant changes")
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                            }
                        } icon: {
                            Image(systemName: "bell.badge")
                                .foregroundColor(.blue)
                        }
                    }
                    
                    // Emergency Alerts
                    Toggle(isOn: $enableEmergencyAlerts) {
                        Label {
                            VStack(alignment: .leading, spacing: 2) {
                                Text("Emergency Alerts")
                                Text("Critical heart rate notifications")
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                            }
                        } icon: {
                            Image(systemName: "exclamationmark.triangle")
                                .foregroundColor(.orange)
                        }
                    }
                    
                    if enableEmergencyAlerts {
                        // High HR Threshold
                        Stepper(value: $emergencyHeartRateThreshold, in: 100...200, step: 5) {
                            HStack {
                                Label("High Heart Rate", systemImage: "arrow.up.heart")
                                    .foregroundColor(.red)
                                Spacer()
                                Text("\(emergencyHeartRateThreshold) bpm")
                                    .foregroundColor(.secondary)
                                    .fontWeight(.medium)
                            }
                        }
                        
                        // Low HR Threshold  
                        Stepper(value: $lowHeartRateThreshold, in: 40...80, step: 5) {
                            HStack {
                                Label("Low Heart Rate", systemImage: "arrow.down.heart")
                                    .foregroundColor(.blue)
                                Spacer()
                                Text("\(lowHeartRateThreshold) bpm")
                                    .foregroundColor(.secondary)
                                    .fontWeight(.medium)
                            }
                        }
                    }
                } header: {
                    Text("Monitoring")
                }
                
                // Data Section
                Section {
                    // Data Summary
                    HStack {
                        Label {
                            VStack(alignment: .leading, spacing: 2) {
                                Text("Stored Records")
                                Text("\(dataManager.healthRecords.count) assessments")
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                            }
                        } icon: {
                            Image(systemName: "heart.text.square")
                                .foregroundColor(.purple)
                        }
                        
                        Spacer()
                        
                        if let lastRecord = dataManager.healthRecords.first,
                           let date = lastRecord.date {
                            Text(date, style: .relative)
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                    }
                    
                    // Export Data
                    Button(action: exportData) {
                        Label {
                            Text("Export Health Data")
                                .foregroundColor(.primary)
                        } icon: {
                            Image(systemName: "square.and.arrow.up")
                                .foregroundColor(.green)
                        }
                    }
                    
                    // Clear Data
                    Button(action: { showingDeleteAlert = true }) {
                        Label {
                            Text("Clear All Data")
                                .foregroundColor(.red)
                        } icon: {
                            Image(systemName: "trash")
                                .foregroundColor(.red)
                        }
                    }
                } header: {
                    Text("Data Management")
                }
                
                // Privacy Section
                Section {
                    // HealthKit Permissions
                    Button(action: openHealthKitSettings) {
                        Label {
                            VStack(alignment: .leading, spacing: 2) {
                                Text("HealthKit Permissions")
                                Text("Manage data access")
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                            }
                        } icon: {
                            Image(systemName: "heart.text.square.fill")
                                .foregroundColor(.pink)
                        }
                    }
                    
                    // Privacy Policy
                    Link(destination: URL(string: "https://www.apple.com/privacy/")!) {
                        Label {
                            Text("Privacy Policy")
                                .foregroundColor(.primary)
                        } icon: {
                            Image(systemName: "hand.raised.fill")
                                .foregroundColor(.blue)
                        }
                    }
                } header: {
                    Text("Privacy")
                }
                
                // About Section
                Section {
                    // App Version
                    HStack {
                        Label("Version", systemImage: "info.circle")
                            .foregroundColor(.gray)
                        Spacer()
                        Text("1.0.0")
                            .foregroundColor(.secondary)
                    }
                    
                    // ML Models Info
                    Button(action: { showingAbout = true }) {
                        Label {
                            VStack(alignment: .leading, spacing: 2) {
                                Text("AI Models")
                                Text("SVM, GBM, CNN • 99.4% accuracy")
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                            }
                        } icon: {
                            Image(systemName: "brain")
                                .foregroundColor(.purple)
                        }
                    }
                    
                    // Developer
                    HStack {
                        Label("Developer", systemImage: "person.circle")
                            .foregroundColor(.gray)
                        Spacer()
                        Text("Rhythm 360 Team")
                            .foregroundColor(.secondary)
                    }
                } header: {
                    Text("About")
                }
            }
            .navigationTitle("Settings")
            .navigationBarTitleDisplayMode(.large)
        }
        .alert("Delete All Data?", isPresented: $showingDeleteAlert) {
            Button("Cancel", role: .cancel) { }
            Button("Delete", role: .destructive) {
                dataManager.deleteAllRecords()
            }
        } message: {
            Text("This will permanently delete all stored health assessments. This action cannot be undone.")
        }
        .sheet(isPresented: $showingAbout) {
            AboutView()
        }
        .alert("Export", isPresented: $showingExportAlert) {
            Button("OK", role: .cancel) { }
        } message: {
            Text(exportAlertMessage)
        }
    }
    
    func exportData() {
        let records = dataManager.healthRecords
        
        // Check if there's data to export
        guard !records.isEmpty else {
            exportAlertMessage = "No health data to export. Start monitoring to collect data."
            showingExportAlert = true
            return
        }
        
        // Create CSV content
        var csvText = "Date,Time,Heart Rate,HRV,Respiratory Rate,Activity,Sleep Quality,Risk Level,Rhythm Status,HRV Pattern\n"
        
        let dateFormatter = DateFormatter()
        dateFormatter.dateFormat = "yyyy-MM-dd"
        let timeFormatter = DateFormatter()
        timeFormatter.dateFormat = "HH:mm:ss"
        
        for record in records {
            let date = record.date ?? Date()
            let dateStr = dateFormatter.string(from: date)
            let timeStr = timeFormatter.string(from: date)
            
            let row = "\(dateStr),\(timeStr),\(Int(record.heartRate)),\(Int(record.hrvMean)),\(Int(record.respiratoryRate)),\(Int(record.activityLevel)),\(Int(record.sleepQuality * 100))%,\(record.riskLevel ?? ""),\(record.rhythmStatus ?? ""),\(record.hrvPattern ?? "")\n"
            csvText.append(row)
        }
        
        // Save to temporary file
        let fileName = "Rhythm360_HealthData_\(dateFormatter.string(from: Date())).csv"
        let path = FileManager.default.temporaryDirectory.appendingPathComponent(fileName)
        
        do {
            try csvText.write(to: path, atomically: true, encoding: .utf8)
            
            // Share the file
            let activityVC = UIActivityViewController(activityItems: [path], applicationActivities: nil)
            
            if let windowScene = UIApplication.shared.connectedScenes.first as? UIWindowScene,
               let rootVC = windowScene.windows.first?.rootViewController {
                // Find the topmost view controller
                var topVC = rootVC
                while let presented = topVC.presentedViewController {
                    topVC = presented
                }
                topVC.present(activityVC, animated: true)
            }
        } catch {
            print("Failed to export CSV: \(error)")
        }
    }
    
    func openHealthKitSettings() {
        if let url = URL(string: UIApplication.openSettingsURLString) {
            UIApplication.shared.open(url)
        }
    }
}

struct AboutView: View {
    @Environment(\.dismiss) private var dismiss
    
    var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 24) {
                    // App Icon and Name
                    VStack(spacing: 12) {
                        Image(systemName: "waveform.path.ecg")
                            .font(.system(size: 60))
                            .foregroundColor(.blue)
                        
                        Text("Rhythm 360")
                            .font(.largeTitle)
                            .fontWeight(.bold)
                        
                        Text("AI-Powered Health Monitoring")
                            .font(.subheadline)
                            .foregroundColor(.secondary)
                    }
                    .padding(.top, 20)
                    
                    // ML Models Section
                    VStack(alignment: .leading, spacing: 16) {
                        Text("Machine Learning Models")
                            .font(.headline)
                        
                        ModelInfoCard(
                            name: "SVM Classifier",
                            accuracy: "92.4%",
                            purpose: "Heart rhythm classification",
                            icon: "waveform.path"
                        )
                        
                        ModelInfoCard(
                            name: "Gradient Boosting",
                            accuracy: "99.4%",
                            purpose: "Health risk assessment",
                            icon: "chart.line.uptrend.xyaxis"
                        )
                        
                        ModelInfoCard(
                            name: "CNN Deep Learning",
                            accuracy: "99.4%",
                            purpose: "HRV pattern detection",
                            icon: "brain"
                        )
                    }
                    .padding(.horizontal)
                    
                    // Features Section
                    VStack(alignment: .leading, spacing: 16) {
                        Text("Features")
                            .font(.headline)
                        
                        FeatureRow(icon: "applewatch", text: "Real-time Apple Watch integration")
                        FeatureRow(icon: "heart.text.square", text: "Comprehensive health analysis")
                        FeatureRow(icon: "chart.line.uptrend.xyaxis", text: "Historical trend tracking")
                        FeatureRow(icon: "bell.badge", text: "Smart health notifications")
                    }
                    .padding(.horizontal)
                    
                    Spacer(minLength: 20)
                }
            }
            .background(Color(UIColor.systemGroupedBackground))
            .navigationTitle("About")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Done") {
                        dismiss()
                    }
                }
            }
        }
    }
}

struct ModelInfoCard: View {
    let name: String
    let accuracy: String
    let purpose: String
    let icon: String
    
    var body: some View {
        HStack {
            Image(systemName: icon)
                .font(.title2)
                .foregroundColor(.blue)
                .frame(width: 40)
            
            VStack(alignment: .leading, spacing: 4) {
                HStack {
                    Text(name)
                        .font(.subheadline)
                        .fontWeight(.medium)
                    Spacer()
                    Text(accuracy)
                        .font(.caption)
                        .fontWeight(.semibold)
                        .foregroundColor(.green)
                }
                Text(purpose)
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
        }
        .padding()
        .background(Color(UIColor.tertiarySystemGroupedBackground))
        .cornerRadius(12)
    }
}

struct FeatureRow: View {
    let icon: String
    let text: String
    
    var body: some View {
        HStack(spacing: 12) {
            Image(systemName: icon)
                .foregroundColor(.blue)
                .frame(width: 24)
            Text(text)
                .font(.subheadline)
            Spacer()
        }
    }
}

struct SettingsView_Previews: PreviewProvider {
    static var previews: some View {
        SettingsView()
    }
}