//
//  HealthTrendsView.swift
//  Rhythm 360
//

import SwiftUI
import Charts

struct HealthTrendsView: View {
    @StateObject private var dataManager = DataManager.shared
    @State private var selectedDays = 7
    
    var trends: HealthTrends {
        dataManager.getHealthTrends(days: selectedDays)
    }
    
    var body: some View {
        ScrollView {
            VStack(spacing: 20) {
                // Days Selector
                Picker("Time Period", selection: $selectedDays) {
                    Text("7 Days").tag(7)
                    Text("30 Days").tag(30)
                    Text("90 Days").tag(90)
                }
                .pickerStyle(.segmented)
                .padding(.horizontal)
                
                // Trend Summary
                VStack(spacing: 16) {
                    TrendCard(
                        title: "Average Heart Rate",
                        value: String(format: "%.0f", trends.averageHeartRate),
                        unit: "bpm",
                        trend: trends.riskTrend,
                        icon: "heart.fill",
                        color: .red
                    )
                    
                    TrendCard(
                        title: "Average HRV",
                        value: String(format: "%.1f", trends.averageHRV),
                        unit: "ms",
                        trend: trends.riskTrend,
                        icon: "waveform.path.ecg",
                        color: .purple
                    )
                    
                    TrendCard(
                        title: "Total Activity",
                        value: String(format: "%.0f", trends.totalActivity),
                        unit: "kcal",
                        trend: .stable,
                        icon: "flame.fill",
                        color: .orange
                    )
                    
                    TrendCard(
                        title: "Sleep Quality",
                        value: String(format: "%.0f%%", trends.averageSleepQuality * 100),
                        unit: "",
                        trend: .stable,
                        icon: "moon.fill",
                        color: .indigo
                    )
                }
                .padding(.horizontal)
                
                // Records Count
                Text("\(trends.recordCount) assessments in selected period")
                    .font(.caption)
                    .foregroundColor(.secondary)
                    .padding()
            }
        }
        .navigationTitle("Health Trends")
        .navigationBarTitleDisplayMode(.inline)
    }
}

struct TrendCard: View {
    let title: String
    let value: String
    let unit: String
    let trend: RiskTrend
    let icon: String
    let color: Color
    
    var trendIcon: String {
        switch trend {
        case .improving: return "arrow.down.circle.fill"
        case .stable: return "equal.circle.fill"
        case .worsening: return "arrow.up.circle.fill"
        }
    }
    
    var trendColor: Color {
        switch trend {
        case .improving: return .green
        case .stable: return .blue
        case .worsening: return .red
        }
    }
    
    var body: some View {
        HStack {
            Image(systemName: icon)
                .font(.title2)
                .foregroundColor(color)
                .frame(width: 50)
            
            VStack(alignment: .leading, spacing: 4) {
                Text(title)
                    .font(.caption)
                    .foregroundColor(.secondary)
                
                HStack {
                    Text(value)
                        .font(.title2)
                        .fontWeight(.bold)
                    
                    if !unit.isEmpty {
                        Text(unit)
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                }
            }
            
            Spacer()
            
            Image(systemName: trendIcon)
                .foregroundColor(trendColor)
                .font(.title3)
        }
        .padding()
        .background(Color.gray.opacity(0.1))
        .cornerRadius(12)
    }
}

struct HistoryView: View {
    @StateObject private var dataManager = DataManager.shared
    
    var body: some View {
        List(dataManager.healthRecords) { record in
            VStack(alignment: .leading, spacing: 8) {
                HStack {
                    Text(record.date ?? Date(), style: .date)
                        .font(.headline)
                    Spacer()
                    Text(record.date ?? Date(), style: .time)
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                
                HStack(spacing: 20) {
                    Label("\(Int(record.heartRate)) bpm", systemImage: "heart.fill")
                        .font(.caption)
                    
                    Label(record.riskLevel ?? "Unknown", systemImage: "shield.fill")
                        .font(.caption)
                        .foregroundColor(record.riskLevel == "Low" ? .green : .orange)
                    
                    Label(record.rhythmStatus ?? "Unknown", systemImage: "waveform.path.ecg")
                        .font(.caption)
                }
            }
            .padding(.vertical, 4)
        }
        .navigationTitle("Assessment History")
        .navigationBarTitleDisplayMode(.inline)
    }
}

struct ExportDataView: View {
    @Environment(\.dismiss) private var dismiss
    
    var body: some View {
        NavigationStack {
            VStack(spacing: 20) {
                Image(systemName: "square.and.arrow.up")
                    .font(.largeTitle)
                    .foregroundColor(.blue)
                
                Text("Export Health Data")
                    .font(.title)
                
                Text("Export your health data as a CSV file to share with healthcare providers")
                    .multilineTextAlignment(.center)
                    .foregroundColor(.secondary)
                    .padding(.horizontal)
                
                Button("Export CSV") {
                    // Export implementation
                    dismiss()
                }
                .padding()
                .background(Color.blue)
                .foregroundColor(.white)
                .cornerRadius(10)
            }
            .padding()
            .navigationBarItems(trailing: Button("Cancel") { dismiss() })
        }
    }
}

struct PrivacyPolicyView: View {
    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                Text("Privacy Policy")
                    .font(.largeTitle)
                    .fontWeight(.bold)
                
                Text("Last updated: \(Date(), style: .date)")
                    .foregroundColor(.secondary)
                
                Section("Data Collection") {
                    Text("Rhythm 360 collects health data from your Apple Watch including heart rate, HRV, respiratory rate, activity levels, and sleep patterns. This data is used exclusively for health monitoring and AI-powered analysis.")
                }
                
                Section("Data Storage") {
                    Text("All health data is stored locally on your device using Core Data with encryption. No data is transmitted to external servers without your explicit consent.")
                }
                
                Section("Data Sharing") {
                    Text("Your health data is never shared with third parties. You have full control over data export for healthcare providers.")
                }
            }
            .padding()
        }
        .navigationBarTitleDisplayMode(.inline)
    }
}

struct TermsOfServiceView: View {
    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                Text("Terms of Service")
                    .font(.largeTitle)
                    .fontWeight(.bold)
                
                Section("Medical Disclaimer") {
                    Text("Rhythm 360 is not a medical device and should not be used for medical diagnosis. Always consult healthcare professionals for medical advice.")
                }
                
                Section("Accuracy") {
                    Text("While our ML models achieve high accuracy (92-99%), results should be verified by medical professionals.")
                }
                
                Section("Liability") {
                    Text("Use of this app is at your own risk. We are not liable for any health decisions made based on app data.")
                }
            }
            .padding()
        }
        .navigationBarTitleDisplayMode(.inline)
    }
}

private struct Section<Content: View>: View {
    let title: String
    let content: Content
    
    init(_ title: String, @ViewBuilder content: () -> Content) {
        self.title = title
        self.content = content()
    }
    
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(title)
                .font(.headline)
            content
                .font(.body)
        }
    }
}