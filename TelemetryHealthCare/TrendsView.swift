//
//  TrendsView.swift
//  Rhythm 360
//
//  Health trends and historical data visualization
//

import SwiftUI
import Charts

struct TrendsView: View {
    @ObservedObject private var dataManager = DataManager.shared
    @State private var selectedTimeRange = TimeRange.week
    @State private var selectedMetric = MetricType.heartRate
    
    enum TimeRange: String, CaseIterable {
        case day = "1D"
        case week = "1W"
        case month = "1M"
        case threeMonths = "3M"
        
        var days: Int {
            switch self {
            case .day: return 1
            case .week: return 7
            case .month: return 30
            case .threeMonths: return 90
            }
        }
    }
    
    enum MetricType: String, CaseIterable {
        case heartRate = "Heart Rate"
        case hrv = "HRV"
        case risk = "Risk Score"
        case activity = "Activity"
        
        var icon: String {
            switch self {
            case .heartRate: return "heart.fill"
            case .hrv: return "waveform.path.ecg"
            case .risk: return "shield.fill"
            case .activity: return "flame.fill"
            }
        }
        
        var color: Color {
            switch self {
            case .heartRate: return .red
            case .hrv: return .purple
            case .risk: return .orange
            case .activity: return .green
            }
        }
    }
    
    var filteredRecords: [HealthRecord] {
        dataManager.fetchRecords(for: selectedTimeRange.days)
    }
    
    var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 20) {
                    // Time Range Selector
                    Picker("Time Range", selection: $selectedTimeRange) {
                        ForEach(TimeRange.allCases, id: \.self) { range in
                            Text(range.rawValue).tag(range)
                        }
                    }
                    .pickerStyle(SegmentedPickerStyle())
                    .padding(.horizontal)
                    
                    // Metric Selector
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
                    
                    // Chart
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
                    
                    // Statistics Cards
                    StatisticsSection(records: filteredRecords, metric: selectedMetric)
                        .padding(.horizontal)
                    
                    // Recent Assessments
                    RecentAssessmentsSection(records: Array(filteredRecords.prefix(5)))
                        .padding(.horizontal)
                    
                    // Bottom padding
                    Color.clear.frame(height: 20)
                }
                .padding(.top)
            }
            .background(Color(UIColor.systemGroupedBackground))
            .navigationTitle("Trends")
            .navigationBarTitleDisplayMode(.large)
        }
        .onAppear {
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