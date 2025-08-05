//
//  ModelManager_CoreML.swift
//  TelemetryHealthCare
//
//  Core ML model integration for health analysis
//

import CoreML
import Foundation

class ModelManager {
    // Load Core ML models
    private var heartRhythmModel: HeartRhythmClassifier?
    private var healthRiskModel: HealthRiskAssessment?
    private var hrvPatternModel: HRVPatternAnalyzer?
    
    init() {
        do {
            // Initialize models
            heartRhythmModel = try HeartRhythmClassifier(configuration: MLModelConfiguration())
            healthRiskModel = try HealthRiskAssessment(configuration: MLModelConfiguration())
            hrvPatternModel = try HRVPatternAnalyzer(configuration: MLModelConfiguration())
            print("✓ All Core ML models loaded successfully")
        } catch {
            print("❌ Error loading models: \(error)")
        }
    }
    
    // MARK: - Heart Rhythm Analysis
    func analyzeHeartRhythm(meanHR: Double, stdHR: Double, pnn50: Double) -> (rhythm: String, confidence: Double) {
        guard let model = heartRhythmModel else {
            return ("Unknown", 0.0)
        }
        
        do {
            let input = HeartRhythmClassifierInput(
                mean_heart_rate: meanHR,
                std_heart_rate: stdHR,
                pnn50: pnn50
            )
            
            let output = try model.prediction(input: input)
            
            // Get the prediction and probability
            let rhythm = output.rhythm_prediction == 1 ? "Irregular" : "Normal"
            let confidence = output.rhythm_probabilityProbability[1] ?? 0.0
            
            return (rhythm, confidence)
        } catch {
            print("Error in rhythm analysis: \(error)")
            return ("Error", 0.0)
        }
    }
    
    // MARK: - Health Risk Assessment
    func assessHealthRisk(
        heartRate: Double,
        hrv: Double,
        respiratoryRate: Double,
        activityLevel: Double,
        sleepQuality: Double
    ) -> (risk: String, confidence: Double) {
        guard let model = healthRiskModel else {
            return ("Unknown", 0.0)
        }
        
        do {
            // Calculate derived features
            let stressIndicator = 1.0 / (1.0 + exp(-0.1 * (heartRate - 75)))
            let hrHrvRatio = heartRate / (hrv + 1)
            let recoveryScore = sleepQuality * hrv / 50
            
            let input = HealthRiskAssessmentInput(
                average_heart_rate: heartRate,
                hrv_mean: hrv,
                respiratory_rate: respiratoryRate,
                activity_level: activityLevel,
                sleep_quality: sleepQuality,
                stress_indicator: stressIndicator,
                hr_hrv_ratio: hrHrvRatio,
                recovery_score: recoveryScore
            )
            
            let output = try model.prediction(input: input)
            
            let risk = output.risk_prediction == 1 ? "High" : "Low"
            let confidence = output.risk_probabilityProbability[Int(output.risk_prediction)] ?? 0.0
            
            return (risk, confidence)
        } catch {
            print("Error in risk assessment: \(error)")
            return ("Error", 0.0)
        }
    }
    
    // MARK: - HRV Pattern Analysis
    func analyzeHRVPattern(rrIntervals: [Double]) -> (pattern: String, confidence: Double) {
        guard let model = hrvPatternModel,
              rrIntervals.count >= 50 else {
            return ("Insufficient Data", 0.0)
        }
        
        do {
            // Calculate features from RR intervals
            let features = calculateHRVFeatures(rrIntervals: rrIntervals)
            
            let input = HRVPatternAnalyzerInput(
                mean_rr: features.meanRR,
                std_rr: features.stdRR,
                min_rr: features.minRR,
                max_rr: features.maxRR,
                q25_rr: features.q25RR,
                q75_rr: features.q75RR,
                mean_diff_rr: features.meanDiffRR,
                std_diff_rr: features.stdDiffRR,
                rmssd: features.rmssd,
                pnn50: features.pnn50,
                low_freq_power: features.lowFreqPower,
                mid_freq_power: features.midFreqPower,
                high_freq_power: features.highFreqPower
            )
            
            let output = try model.prediction(input: input)
            
            // Map prediction to pattern name
            let patterns = ["Normal", "AFib", "Bradycardia", "Tachycardia"]
            let patternIndex = Int(output.pattern_prediction)
            let pattern = patterns[safe: patternIndex] ?? "Unknown"
            
            // Get confidence from probabilities
            let confidence = output.pattern_probabilities[patternIndex] ?? 0.0
            
            return (pattern, confidence)
        } catch {
            print("Error in HRV analysis: \(error)")
            return ("Error", 0.0)
        }
    }
    
    // MARK: - Helper Functions
    private func calculateHRVFeatures(rrIntervals: [Double]) -> HRVFeatures {
        let sorted = rrIntervals.sorted()
        let diffs = zip(rrIntervals.dropFirst(), rrIntervals).map { $0 - $1 }
        
        // Basic statistics
        let meanRR = rrIntervals.reduce(0, +) / Double(rrIntervals.count)
        let stdRR = sqrt(rrIntervals.map { pow($0 - meanRR, 2) }.reduce(0, +) / Double(rrIntervals.count))
        
        // RMSSD
        let rmssd = sqrt(diffs.map { pow($0, 2) }.reduce(0, +) / Double(diffs.count))
        
        // pNN50
        let nn50 = diffs.filter { abs($0) > 50 }.count
        let pnn50 = Double(nn50) / Double(diffs.count)
        
        // Simplified frequency domain (would need FFT for real implementation)
        let lowFreqPower = stdRR * 0.5
        let midFreqPower = stdRR * 0.3
        let highFreqPower = stdRR * 0.2
        
        return HRVFeatures(
            meanRR: meanRR,
            stdRR: stdRR,
            minRR: sorted.first ?? 0,
            maxRR: sorted.last ?? 0,
            q25RR: sorted[sorted.count / 4],
            q75RR: sorted[3 * sorted.count / 4],
            meanDiffRR: diffs.reduce(0, +) / Double(diffs.count),
            stdDiffRR: sqrt(diffs.map { pow($0 - (diffs.reduce(0, +) / Double(diffs.count)), 2) }.reduce(0, +) / Double(diffs.count)),
            rmssd: rmssd,
            pnn50: pnn50,
            lowFreqPower: lowFreqPower,
            midFreqPower: midFreqPower,
            highFreqPower: highFreqPower
        )
    }
}

// MARK: - Supporting Types
struct HRVFeatures {
    let meanRR: Double
    let stdRR: Double
    let minRR: Double
    let maxRR: Double
    let q25RR: Double
    let q75RR: Double
    let meanDiffRR: Double
    let stdDiffRR: Double
    let rmssd: Double
    let pnn50: Double
    let lowFreqPower: Double
    let midFreqPower: Double
    let highFreqPower: Double
}

// Safe array access
extension Array {
    subscript(safe index: Int) -> Element? {
        return index >= 0 && index < count ? self[index] : nil
    }
}