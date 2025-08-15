//
//  TrendsView.swift
//  TelemetryHealthCare
//
//  Comprehensive health trends analysis and data visualization interface.
//
//  This view provides powerful data visualization and trend analysis capabilities,
//  allowing users to understand their health patterns over time. Features include:
//
//  - Interactive time range selection (1 day to 3 months)
//  - Multi-metric visualization (heart rate, HRV, risk scores, activity)
//  - Statistical analysis with trend calculations
//  - Recent assessment history
//  - Responsive chart design using Swift Charts
//
//  The interface adapts to different data densities and provides meaningful
//  insights even with limited historical data.
//
//  Created by TelemetryHealthCare Team on 2024.
//

import SwiftUI
import Charts

/// Advanced health trends visualization and analysis view
///
/// Provides comprehensive historical health data analysis with interactive
/// charts, statistical summaries, and trend identification. Supports multiple
/// time ranges and health metrics for detailed pattern recognition.
///
/// - Note: Requires historical health data from DataManager
/// - Important: Charts update automatically when new data is available
struct TrendsView: View {
    /// Core Data manager for historical health records
    @ObservedObject private var dataManager = DataManager.shared
    
    /// Currently selected time range for analysis
    @State private var selectedTimeRange = TimeRange.week
    
    /// Currently selected health metric for visualization
    @State private var selectedMetric = MetricType.heartRate
    
    // MARK: - Time Range Configuration
    
    /// Available time ranges for health data analysis
    enum TimeRange: String, CaseIterable {
        case day = "1D"            // Last 24 hours
        case week = "1W"           // Last 7 days
        case month = "1M"          // Last 30 days
        case threeMonths = "3M"    // Last 90 days
        
        /// Converts time range to number of days for data queries
        var days: Int {
            switch self {
            case .day: return 1
            case .week: return 7
            case .month: return 30
            case .threeMonths: return 90
            }
        }
    }
    
    // MARK: - Health Metrics Configuration
    
    /// Available health metrics for trend visualization
    enum MetricType: String, CaseIterable {
        case heartRate = "Heart Rate"  // BPM measurements
        case hrv = "HRV"               // Heart rate variability
        case risk = "Risk Score"       // ML risk assessments
        case activity = "Activity"     // Daily activity levels
        
        /// SF Symbol icon for metric visualization
        var icon: String {
            switch self {
            case .heartRate: return "heart.fill"
            case .hrv: return "waveform.path.ecg"
            case .risk: return "shield.fill"
            case .activity: return "flame.fill"
            }
        }
        
        /// Color theme for metric visualization
        var color: Color {
            switch self {
            case .heartRate: return .red      // Medical standard for heart rate
            case .hrv: return .purple         // Distinctive for HRV analysis
            case .risk: return .orange        // Warning color for risk
            case .activity: return .green     // Positive association with activity
            }
        }
    }
    
    /// Computed property providing filtered health records for selected time range
    var filteredRecords: [HealthRecord] {
        dataManager.fetchRecords(for: selectedTimeRange.days)
    }
    
    // MARK: - Main View Interface
    
    var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 20) {
                    // MARK: Time Range Selection
                    // Segmented control for choosing analysis period
                    Picker("Time Range", selection: $selectedTimeRange) {
                        ForEach(TimeRange.allCases, id: \.self) { range in
                            Text(range.rawValue).tag(range)
                        }
                    }
                    .pickerStyle(SegmentedPickerStyle())
                    .padding(.horizontal)
                    
                    // MARK: Metric Selection
                    // Horizontal scroll view for metric type selection
                    ScrollView(.horizontal, showsIndicators: false) {
                        HStack(spacing: 12) {
                            ForEach(MetricType.allCases, id: \.self) { metric in
                                MetricButton(
                                    metric: metric,
                                    isSelected: selectedMetric == metric
                                ) {
                                    selectedMetric = metric
                                }
                            }
                        }
                        .padding(.horizontal)
                    }
                    
                    // MARK: Data Visualization
                    // Main chart area with conditional empty state
                    if !filteredRecords.isEmpty {
                        ChartView(
                            records: filteredRecords,
                            metric: selectedMetric
                        )
                        .frame(height: 250)
                        .padding(.horizontal)
                    } else {
                        EmptyChartView()
                            .frame(height: 250)
                            .padding(.horizontal)
                    }
                    
                    // MARK: Statistical Analysis
                    // Summary statistics for selected metric and time range
                    StatisticsSection(records: filteredRecords, metric: selectedMetric)
                        .padding(.horizontal)
                    
                    // MARK: Assessment History
                    // Recent health assessments for context
                    RecentAssessmentsSection(records: Array(filteredRecords.prefix(5)))
                        .padding(.horizontal)
                    
                    // Bottom padding for tab bar clearance
                    Color.clear.frame(height: 20)
                }
                .padding(.top)
            }
            .background(Color(UIColor.systemGroupedBackground))
            .navigationTitle("Trends")
            .navigationBarTitleDisplayMode(.large)
        }
        .onAppear {
            // Refresh health records when view appears
            dataManager.fetchHealthRecords()
        }
    }
}

struct MetricButton: View {
    let metric: TrendsView.MetricType
    let isSelected: Bool
    let action: () -> Void
    
    var body: some View {
        Button(action: action) {
            HStack(spacing: 6) {
                Image(systemName: metric.icon)
                    .font(.caption)
                Text(metric.rawValue)
                    .font(.caption)
                    .fontWeight(.medium)
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 8)
            .background(isSelected ? metric.color : Color(UIColor.tertiarySystemGroupedBackground))
            .foregroundColor(isSelected ? .white : .primary)
            .cornerRadius(20)
        }
    }
}

struct ChartView: View {
    let records: [HealthRecord]
    let metric: TrendsView.MetricType
    
    var chartData: [(date: Date, value: Double)] {
        records.compactMap { record in
            guard let date = record.date else { return nil }
            
            let value: Double
            switch metric {
            case .heartRate:
                value = record.heartRate
            case .hrv:
                value = record.hrvMean
            case .risk:
                value = record.riskLevel == "High" ? 1.0 : 0.0
            case .activity:
                value = record.activityLevel
            }
            
            return (date, value)
        }
    }
    
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            // Chart Header
            HStack {
                Image(systemName: metric.icon)
                    .foregroundColor(metric.color)
                Text(metric.rawValue)
                    .font(.headline)
                Spacer()
            }
            
            // Chart
            Chart(chartData, id: \.date) { item in
                LineMark(
                    x: .value("Date", item.date),
                    y: .value("Value", item.value)
                )
                .foregroundStyle(metric.color)
                .lineStyle(StrokeStyle(lineWidth: 2, lineCap: .round, lineJoin: .round))
                
                AreaMark(
                    x: .value("Date", item.date),
                    y: .value("Value", item.value)
                )
                .foregroundStyle(
                    LinearGradient(
                        colors: [metric.color.opacity(0.3), metric.color.opacity(0.05)],
                        startPoint: .top,
                        endPoint: .bottom
                    )
                )
            }
            .chartXAxis {
                AxisMarks(values: .automatic(desiredCount: 4)) { _ in
                    AxisGridLine(stroke: StrokeStyle(lineWidth: 0.5, dash: [2, 2]))
                        .foregroundStyle(Color.secondary.opacity(0.3))
                    AxisValueLabel()
                        .font(.caption2)
                }
            }
            .chartYAxis {
                AxisMarks(position: .leading, values: .automatic(desiredCount: 5)) { _ in
                    AxisGridLine(stroke: StrokeStyle(lineWidth: 0.5, dash: [2, 2]))
                        .foregroundStyle(Color.secondary.opacity(0.3))
                    AxisValueLabel()
                        .font(.caption2)
                }
            }
        }
        .padding()
        .background(Color(UIColor.tertiarySystemGroupedBackground))
        .cornerRadius(16)
    }
}

struct EmptyChartView: View {
    var body: some View {
        VStack {
            Image(systemName: "chart.line.uptrend.xyaxis")
                .font(.largeTitle)
                .foregroundColor(.secondary)
            Text("No data for selected period")
                .font(.caption)
                .foregroundColor(.secondary)
        }
        .frame(maxWidth: .infinity)
        .padding()
        .background(Color(UIColor.tertiarySystemGroupedBackground))
        .cornerRadius(16)
    }
}

struct StatisticsSection: View {
    let records: [HealthRecord]
    let metric: TrendsView.MetricType
    
    var statistics: (min: Double, max: Double, avg: Double, trend: String) {
        guard !records.isEmpty else { return (0, 0, 0, "—") }
        
        let values: [Double]
        switch metric {
        case .heartRate:
            values = records.map { $0.heartRate }
        case .hrv:
            values = records.map { $0.hrvMean }
        case .risk:
            values = records.map { $0.riskLevel == "High" ? 1.0 : 0.0 }
        case .activity:
            values = records.map { $0.activityLevel }
        }
        
        let min = values.min() ?? 0
        let max = values.max() ?? 0
        let avg = values.reduce(0, +) / Double(values.count)
        
        // Calculate trend
        let trend: String
        if values.count > 1 {
            let firstHalf = Array(values.prefix(values.count / 2))
            let secondHalf = Array(values.suffix(values.count / 2))
            let firstAvg = firstHalf.reduce(0, +) / Double(firstHalf.count)
            let secondAvg = secondHalf.reduce(0, +) / Double(secondHalf.count)
            
            if secondAvg > firstAvg * 1.05 {
                trend = "↑"
            } else if secondAvg < firstAvg * 0.95 {
                trend = "↓"
            } else {
                trend = "→"
            }
        } else {
            trend = "—"
        }
        
        return (min, max, avg, trend)
    }
    
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Statistics")
                .font(.headline)
            
            HStack(spacing: 12) {
                StatCard(label: "Min", value: formatValue(statistics.min), color: .blue)
                StatCard(label: "Avg", value: formatValue(statistics.avg), color: .green)
                StatCard(label: "Max", value: formatValue(statistics.max), color: .orange)
                StatCard(label: "Trend", value: statistics.trend, color: .purple)
            }
        }
    }
    
    func formatValue(_ value: Double) -> String {
        switch metric {
        case .heartRate:
            return "\(Int(value))"
        case .hrv:
            return String(format: "%.0f", value)
        case .risk:
            return value > 0.5 ? "High" : "Low"
        case .activity:
            return "\(Int(value))"
        }
    }
}

struct StatCard: View {
    let label: String
    let value: String
    let color: Color
    
    var body: some View {
        VStack(spacing: 4) {
            Text(label)
                .font(.caption2)
                .foregroundColor(.secondary)
            Text(value)
                .font(.title3)
                .fontWeight(.semibold)
                .foregroundColor(color)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 12)
        .background(Color(UIColor.tertiarySystemGroupedBackground))
        .cornerRadius(12)
    }
}

struct RecentAssessmentsSection: View {
    let records: [HealthRecord]
    
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Recent Assessments")
                .font(.headline)
            
            if records.isEmpty {
                Text("No recent assessments")
                    .font(.caption)
                    .foregroundColor(.secondary)
                    .frame(maxWidth: .infinity)
                    .padding()
                    .background(Color(UIColor.tertiarySystemGroupedBackground))
                    .cornerRadius(12)
            } else {
                VStack(spacing: 8) {
                    ForEach(records, id: \.self) { record in
                        AssessmentRow(record: record)
                    }
                }
            }
        }
    }
}

struct AssessmentRow: View {
    let record: HealthRecord
    
    var body: some View {
        HStack {
            VStack(alignment: .leading, spacing: 4) {
                Text(record.date ?? Date(), style: .date)
                    .font(.caption)
                    .fontWeight(.medium)
                
                HStack(spacing: 12) {
                    Label("\(Int(record.heartRate)) bpm", systemImage: "heart.fill")
                        .font(.caption2)
                        .foregroundColor(.secondary)
                    
                    Label(record.riskLevel ?? "Unknown", systemImage: "shield.fill")
                        .font(.caption2)
                        .foregroundColor(record.riskLevel == "Low" ? .green : .orange)
                }
            }
            
            Spacer()
            
            Text(record.date ?? Date(), style: .time)
                .font(.caption2)
                .foregroundColor(.secondary)
        }
        .padding()
        .background(Color(UIColor.tertiarySystemGroupedBackground))
        .cornerRadius(12)
    }
}

struct TrendsView_Previews: PreviewProvider {
    static var previews: some View {
        TrendsView()
    }
}