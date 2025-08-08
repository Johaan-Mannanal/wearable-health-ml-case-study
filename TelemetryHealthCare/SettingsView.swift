//
//  SettingsView.swift
//  Rhythm 360
//
//  Settings and preferences management
//

import SwiftUI

struct SettingsView: View {
    @AppStorage("monitoringInterval") private var monitoringInterval = 5
    @AppStorage("enableNotifications") private var enableNotifications = true
    @AppStorage("enableEmergencyAlerts") private var enableEmergencyAlerts = false
    @AppStorage("emergencyHeartRateThreshold") private var emergencyHeartRateThreshold = 120
    @AppStorage("autoExportData") private var autoExportData = false
    @StateObject private var dataManager = DataManager.shared
    @State private var showingDeleteAlert = false
    @State private var showingExportSheet = false
    @Environment(\.dismiss) private var dismiss
    
    var body: some View {
        NavigationStack {
            Form {
                // Monitoring Settings
                Section {
                    Picker("Update Interval", selection: $monitoringInterval) {
                        Text("5 seconds").tag(5)
                        Text("10 seconds").tag(10)
                        Text("30 seconds").tag(30)
                        Text("1 minute").tag(60)
                    }
                    
                    Toggle("Enable Notifications", isOn: $enableNotifications)
                    
                    Toggle("Emergency Alerts", isOn: $enableEmergencyAlerts)
                    
                    if enableEmergencyAlerts {
                        Stepper("High HR Alert: \(emergencyHeartRateThreshold) bpm", 
                               value: $emergencyHeartRateThreshold, 
                               in: 100...200, 
                               step: 5)
                    }
                } header: {
                    Text("Monitoring")
                } footer: {
                    Text("Adjust how frequently the app checks your health data")
                }
                
                // Data Management
                Section {
                    HStack {
                        Text("Stored Records")
                        Spacer()
                        Text("\(dataManager.healthRecords.count)")
                            .foregroundColor(.secondary)
                    }
                    
                    Button(action: { showingExportSheet = true }) {
                        Label("Export Health Data", systemImage: "square.and.arrow.up")
                    }
                    
                    Toggle("Auto-Export Weekly", isOn: $autoExportData)
                    
                    Button(role: .destructive, action: { showingDeleteAlert = true }) {
                        Label("Clear All Data", systemImage: "trash")
                            .foregroundColor(.red)
                    }
                } header: {
                    Text("Data Management")
                }
                
                // Health Trends
                Section {
                    NavigationLink(destination: HealthTrendsView()) {
                        Label("View Health Trends", systemImage: "chart.line.uptrend.xyaxis")
                    }
                    
                    NavigationLink(destination: HistoryView()) {
                        Label("Assessment History", systemImage: "clock.arrow.circlepath")
                    }
                } header: {
                    Text("Analytics")
                }
                
                // About
                Section {
                    HStack {
                        Text("Version")
                        Spacer()
                        Text("1.0.0")
                            .foregroundColor(.secondary)
                    }
                    
                    NavigationLink(destination: PrivacyPolicyView()) {
                        Label("Privacy Policy", systemImage: "hand.raised")
                    }
                    
                    NavigationLink(destination: TermsOfServiceView()) {
                        Label("Terms of Service", systemImage: "doc.text")
                    }
                    
                    Link(destination: URL(string: "mailto:support@rhythm360.health")!) {
                        Label("Contact Support", systemImage: "envelope")
                    }
                } header: {
                    Text("About")
                }
            }
            .navigationTitle("Settings")
            .navigationBarTitleDisplayMode(.large)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Done") {
                        dismiss()
                    }
                }
            }
            .alert("Delete All Data?", isPresented: $showingDeleteAlert) {
                Button("Cancel", role: .cancel) { }
                Button("Delete", role: .destructive) {
                    dataManager.deleteAllRecords()
                }
            } message: {
                Text("This will permanently delete all stored health assessments. This action cannot be undone.")
            }
            .sheet(isPresented: $showingExportSheet) {
                ExportDataView()
            }
        }
    }
}

struct SettingsView_Previews: PreviewProvider {
    static var previews: some View {
        SettingsView()
    }
}