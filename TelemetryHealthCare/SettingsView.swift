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
                        HStack {
                            Label("High Heart Rate", systemImage: "arrow.up.heart")
                                .foregroundColor(.red)
                            Spacer()
                            Text("\(emergencyHeartRateThreshold) bpm")
                                .foregroundColor(.secondary)
                        }
                        .contentShape(Rectangle())
                        .onTapGesture {
                            // Could show a picker here
                        }
                        
                        // Low HR Threshold
                        HStack {
                            Label("Low Heart Rate", systemImage: "arrow.down.heart")
                                .foregroundColor(.blue)
                            Spacer()
                            Text("\(lowHeartRateThreshold) bpm")
                                .foregroundColor(.secondary)
                        }
                        .contentShape(Rectangle())
                        .onTapGesture {
                            // Could show a picker here
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
    }
    
    func exportData() {
        // Implementation for data export
        print("Exporting data...")
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