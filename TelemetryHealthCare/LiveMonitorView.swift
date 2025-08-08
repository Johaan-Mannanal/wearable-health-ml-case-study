//
//  LiveMonitorView.swift
//  Rhythm 360
//
//  Live heart rate monitoring with real-time visualization
//

import SwiftUI
import Charts

struct LiveMonitorView: View {
    @StateObject private var liveManager = LiveHeartRateManager.shared
    @State private var pulseScale: CGFloat = 1.0
    @State private var showingWorkoutControls = false
    @Environment(\.dismiss) private var dismiss
    
    var body: some View {
        NavigationStack {
            ZStack {
                // Gradient background
                LinearGradient(
                    colors: [
                        Color.black,
                        liveManager.currentZone.color.opacity(0.3)
                    ],
                    startPoint: .top,
                    endPoint: .bottom
                )
                .ignoresSafeArea()
                
                ScrollView {
                    VStack(spacing: 20) {
                        // Live Heart Rate Display
                        LiveHeartRateCard(
                            heartRate: liveManager.currentHeartRate,
                            zone: liveManager.currentZone,
                            pulseScale: $pulseScale
                        )
                        
                        // Real-time Chart
                        LiveChartView(readings: liveManager.recentReadings)
                            .frame(height: 200)
                            .padding(.horizontal)
                        
                        // Zone Indicator
                        HeartRateZoneIndicator(
                            currentZone: liveManager.currentZone,
                            isWorkout: liveManager.isWorkoutActive
                        )
                        .padding(.horizontal)
                        
                        // Stats Grid
                        LiveStatsGrid(readings: liveManager.recentReadings)
                            .padding(.horizontal)
                        
                        // Workout Controls
                        WorkoutControlsCard(
                            isWorkoutActive: liveManager.isWorkoutActive,
                            startWorkout: { liveManager.startWorkoutSession() },
                            endWorkout: { liveManager.endWorkoutSession() }
                        )
                        .padding(.horizontal)
                        
                        // Alert Status
                        if liveManager.alertStatus != .normal {
                            AlertStatusCard(status: liveManager.alertStatus)
                                .padding(.horizontal)
                        }
                    }
                    .padding(.vertical)
                }
            }
            .navigationTitle("Live Monitor")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    Button("Close") {
                        liveManager.stopLiveMonitoring()
                        dismiss()
                    }
                }
                
                ToolbarItem(placement: .navigationBarTrailing) {
                    Image(systemName: liveManager.isLiveMonitoring ? "dot.radiowaves.left.and.right" : "antenna.radiowaves.left.and.right.slash")
                        .foregroundColor(liveManager.isLiveMonitoring ? .green : .red)
                        .symbolEffect(.pulse, value: liveManager.isLiveMonitoring)
                }
            }
        }
        .onAppear {
            liveManager.startLiveMonitoring()
            startPulseAnimation()
        }
        .onDisappear {
            liveManager.stopLiveMonitoring()
        }
    }
    
    private func startPulseAnimation() {
        withAnimation(.easeInOut(duration: 60.0 / max(liveManager.currentHeartRate, 60)).repeatForever(autoreverses: true)) {
            pulseScale = 1.15
        }
    }
}

// MARK: - Live Heart Rate Card
struct LiveHeartRateCard: View {
    let heartRate: Double
    let zone: HeartRateZone
    @Binding var pulseScale: CGFloat
    
    var body: some View {
        VStack(spacing: 16) {
            // Animated Heart
            ZStack {
                // Pulse rings
                ForEach(0..<3) { index in
                    Circle()
                        .stroke(zone.color.opacity(0.3 - Double(index) * 0.1), lineWidth: 2)
                        .frame(width: 120 + CGFloat(index * 30), height: 120 + CGFloat(index * 30))
                        .scaleEffect(pulseScale)
                        .animation(
                            .easeInOut(duration: 60.0 / max(heartRate, 60))
                            .repeatForever(autoreverses: true)
                            .delay(Double(index) * 0.1),
                            value: pulseScale
                        )
                }
                
                // Heart icon
                Image(systemName: "heart.fill")
                    .font(.system(size: 60))
                    .foregroundColor(zone.color)
                    .scaleEffect(pulseScale)
            }
            
            // BPM Display
            VStack(spacing: 4) {
                Text("\(Int(heartRate))")
                    .font(.system(size: 72, weight: .bold, design: .rounded))
                    .foregroundColor(.white)
                
                Text("BPM")
                    .font(.title3)
                    .foregroundColor(.white.opacity(0.7))
                
                Text(zone.rawValue)
                    .font(.headline)
                    .foregroundColor(zone.color)
                    .padding(.horizontal, 12)
                    .padding(.vertical, 4)
                    .background(zone.color.opacity(0.2))
                    .cornerRadius(20)
            }
        }
        .padding(.vertical, 30)
    }
}

// MARK: - Live Chart View
struct LiveChartView: View {
    let readings: [HeartRateReading]
    
    var body: some View {
        Chart(readings.suffix(60)) { reading in
            LineMark(
                x: .value("Time", reading.timestamp),
                y: .value("BPM", reading.value)
            )
            .foregroundStyle(
                .linearGradient(
                    colors: [.red, .orange, .yellow],
                    startPoint: .bottom,
                    endPoint: .top
                )
            )
            .lineStyle(StrokeStyle(lineWidth: 3, lineCap: .round, lineJoin: .round))
            
            AreaMark(
                x: .value("Time", reading.timestamp),
                y: .value("BPM", reading.value)
            )
            .foregroundStyle(
                .linearGradient(
                    colors: [.red.opacity(0.1), .red.opacity(0.3)],
                    startPoint: .bottom,
                    endPoint: .top
                )
            )
        }
        .chartYScale(domain: 40...180)
        .chartXAxis {
            AxisMarks(values: .automatic(desiredCount: 4)) { _ in
                AxisGridLine(stroke: StrokeStyle(lineWidth: 0.5))
                    .foregroundStyle(Color.white.opacity(0.2))
            }
        }
        .chartYAxis {
            AxisMarks(position: .leading) { value in
                AxisGridLine(stroke: StrokeStyle(lineWidth: 0.5))
                    .foregroundStyle(Color.white.opacity(0.2))
                AxisValueLabel()
                    .foregroundStyle(Color.white.opacity(0.7))
            }
        }
        .background(Color.black.opacity(0.3))
        .cornerRadius(16)
    }
}

// MARK: - Heart Rate Zone Indicator
struct HeartRateZoneIndicator: View {
    let currentZone: HeartRateZone
    let isWorkout: Bool
    
    var zones: [HeartRateZone] {
        isWorkout ? [.warmup, .fatBurn, .cardio, .peak, .maximum] :
                   [.low, .resting, .normal, .elevated, .high]
    }
    
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(isWorkout ? "Exercise Zones" : "Heart Rate Zones")
                .font(.headline)
                .foregroundColor(.white)
            
            HStack(spacing: 4) {
                ForEach(zones, id: \.self) { zone in
                    VStack(spacing: 4) {
                        RoundedRectangle(cornerRadius: 4)
                            .fill(zone.color)
                            .frame(height: currentZone == zone ? 40 : 30)
                            .overlay(
                                currentZone == zone ?
                                Image(systemName: "chevron.down")
                                    .foregroundColor(.white)
                                    .offset(y: -25) : nil
                            )
                        
                        Text(zone.rawValue)
                            .font(.system(size: 10))
                            .foregroundColor(currentZone == zone ? .white : .white.opacity(0.5))
                    }
                }
            }
        }
        .padding()
        .background(Color.black.opacity(0.3))
        .cornerRadius(16)
    }
}

// MARK: - Live Stats Grid
struct LiveStatsGrid: View {
    let readings: [HeartRateReading]
    
    var stats: (min: Double, max: Double, avg: Double, variability: Double) {
        guard !readings.isEmpty else { return (0, 0, 0, 0) }
        
        let values = readings.map { $0.value }
        let min = values.min() ?? 0
        let max = values.max() ?? 0
        let avg = values.reduce(0, +) / Double(values.count)
        
        let variance = values.map { pow($0 - avg, 2) }.reduce(0, +) / Double(values.count)
        let std = sqrt(variance)
        
        return (min, max, avg, std)
    }
    
    var body: some View {
        LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible())], spacing: 12) {
            StatCard(title: "Minimum", value: "\(Int(stats.min))", unit: "bpm", color: .blue)
            StatCard(title: "Maximum", value: "\(Int(stats.max))", unit: "bpm", color: .red)
            StatCard(title: "Average", value: "\(Int(stats.avg))", unit: "bpm", color: .green)
            StatCard(title: "Variability", value: String(format: "%.1f", stats.variability), unit: "std", color: .purple)
        }
    }
}

struct StatCard: View {
    let title: String
    let value: String
    let unit: String
    let color: Color
    
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(title)
                .font(.caption)
                .foregroundColor(.white.opacity(0.7))
            
            HStack(alignment: .lastTextBaseline, spacing: 4) {
                Text(value)
                    .font(.title2)
                    .fontWeight(.bold)
                    .foregroundColor(.white)
                
                Text(unit)
                    .font(.caption)
                    .foregroundColor(.white.opacity(0.5))
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding()
        .background(color.opacity(0.2))
        .cornerRadius(12)
    }
}

// MARK: - Workout Controls Card
struct WorkoutControlsCard: View {
    let isWorkoutActive: Bool
    let startWorkout: () -> Void
    let endWorkout: () -> Void
    
    var body: some View {
        VStack(spacing: 12) {
            Text("Workout Mode")
                .font(.headline)
                .foregroundColor(.white)
            
            if isWorkoutActive {
                Button(action: endWorkout) {
                    Label("End Workout", systemImage: "stop.circle.fill")
                        .frame(maxWidth: .infinity)
                        .padding()
                        .background(Color.red)
                        .foregroundColor(.white)
                        .cornerRadius(12)
                }
            } else {
                Button(action: startWorkout) {
                    Label("Start Workout", systemImage: "figure.run")
                        .frame(maxWidth: .infinity)
                        .padding()
                        .background(Color.green)
                        .foregroundColor(.white)
                        .cornerRadius(12)
                }
            }
            
            Text(isWorkoutActive ? "Enhanced monitoring active" : "Tap to enable workout tracking")
                .font(.caption)
                .foregroundColor(.white.opacity(0.7))
        }
        .padding()
        .background(Color.black.opacity(0.3))
        .cornerRadius(16)
    }
}

// MARK: - Alert Status Card
struct AlertStatusCard: View {
    let status: AlertStatus
    
    var statusColor: Color {
        switch status {
        case .normal: return .green
        case .monitoring: return .yellow
        case .warning: return .orange
        case .critical: return .red
        }
    }
    
    var statusText: String {
        switch status {
        case .normal: return "Normal"
        case .monitoring: return "Monitoring Irregularity"
        case .warning: return "Warning - Check Values"
        case .critical: return "Critical - Seek Attention"
        }
    }
    
    var body: some View {
        HStack {
            Image(systemName: status == .critical ? "exclamationmark.triangle.fill" : "exclamationmark.circle.fill")
                .foregroundColor(statusColor)
            
            Text(statusText)
                .font(.headline)
                .foregroundColor(.white)
            
            Spacer()
        }
        .padding()
        .background(statusColor.opacity(0.2))
        .overlay(
            RoundedRectangle(cornerRadius: 12)
                .stroke(statusColor, lineWidth: 2)
        )
        .cornerRadius(12)
    }
}

struct LiveMonitorView_Previews: PreviewProvider {
    static var previews: some View {
        LiveMonitorView()
    }
}