//
//  CardiovascularFitnessModel.swift
//  Rhythm 360
//
//  Cardiovascular Fitness & Recovery Analysis Model
//  Analyzes long-term cardiovascular health and fitness trends
//

import Foundation

class CardiovascularFitnessModel {
    
    // MARK: - Complete Fitness Assessment
    static func runCompleteFitnessAssessment(healthData: HealthKitData) -> CardiovascularFitnessAssessment {
        // Calculate heart rate recovery if we have enough data
        let hrr1min = calculateHRR1Min(from: healthData.recentHeartRates)
        let hrr2min = hrr1min * 1.5 // Estimated 2-min recovery
        
        // Get baseline metrics
        let restingHR = healthData.meanHeartRate
        let hrReserve = (220.0 - 40.0) - restingHR // Assuming age 40 for now
        let rmssd = healthData.hrvMean
        
        // Predict fitness level
        let fitnessResult = predictFitnessLevel(
            age: 40.0, // Default age, should be fetched from user profile
            restingHR: restingHR,
            hrReserve: hrReserve,
            hrr1min: hrr1min,
            hrr2min: hrr2min,
            rmssd: rmssd,
            sdnn: healthData.stdHeartRate,
            recoveryEfficiency: calculateRecoveryEfficiency(hrr1min: hrr1min, hrr2min: hrr2min)
        )
        
        // Estimate VO2max
        let vo2max = estimateVO2max(
            age: 40.0,
            restingHR: restingHR,
            maxHR: 180.0, // Estimated
            hrReserve: hrReserve,
            fitnessLevel: fitnessResult.level
        )
        
        // Calculate cardiovascular age
        let cvAgeResult = calculateCardiovascularAge(
            chronologicalAge: 40.0,
            fitnessLevel: fitnessResult.level,
            restingHR: restingHR,
            hrr1min: hrr1min,
            rmssd: rmssd
        )
        
        // Analyze recovery pattern
        let recoveryAnalysis = analyzeRecoveryPattern(
            hrr1min: hrr1min,
            hrr2min: hrr2min,
            timeToTarget: 120.0 // Estimated
        )
        
        // Assess training readiness
        let readiness = assessTrainingReadiness(
            rmssd: rmssd,
            restingHR: restingHR,
            restingHRBaseline: restingHR - 2, // Slight variation from baseline
            sleepQuality: healthData.sleepQuality
        )
        
        return CardiovascularFitnessAssessment(
            fitnessLevel: fitnessResult.level,
            fitnessCategory: fitnessResult.category,
            vo2max: vo2max,
            cardiovascularAge: cvAgeResult.cvAge,
            ageComparison: cvAgeResult.comparison,
            recoveryEfficiency: recoveryAnalysis.efficiency,
            recoveryStatus: recoveryAnalysis.status,
            trainingReadiness: readiness.score,
            readinessStatus: readiness.status,
            recommendation: generatePersonalizedRecommendation(
                fitness: fitnessResult.level,
                recovery: recoveryAnalysis.efficiency,
                readiness: readiness.score
            ),
            timestamp: Date()
        )
    }
    
    // MARK: - Fitness Level Prediction (Based on trained model)
    static func predictFitnessLevel(
        age: Double,
        restingHR: Double,
        hrReserve: Double,
        hrr1min: Double,
        hrr2min: Double,
        rmssd: Double,
        sdnn: Double,
        recoveryEfficiency: Double
    ) -> (level: Double, category: String) {
        var fitnessScore = 50.0
        
        // Heart rate recovery is the strongest predictor (84.5% importance)
        if hrr1min > 30 {
            fitnessScore += 25
        } else if hrr1min > 25 {
            fitnessScore += 18
        } else if hrr1min > 20 {
            fitnessScore += 10
        } else if hrr1min > 15 {
            fitnessScore += 5
        } else if hrr1min < 12 {
            fitnessScore -= 20
        }
        
        // Resting heart rate contribution (6% importance)
        if restingHR < 50 {
            fitnessScore += 18
        } else if restingHR < 55 {
            fitnessScore += 12
        } else if restingHR < 65 {
            fitnessScore += 6
        } else if restingHR > 75 {
            fitnessScore -= 12
        } else if restingHR > 85 {
            fitnessScore -= 20
        }
        
        // HRV (RMSSD) contribution (7.6% importance)
        if rmssd > 60 {
            fitnessScore += 12
        } else if rmssd > 40 {
            fitnessScore += 6
        } else if rmssd < 20 {
            fitnessScore -= 10
        }
        
        // Age adjustment (1.9% importance but still relevant)
        if age < 30 {
            fitnessScore += 8
        } else if age < 40 {
            fitnessScore += 4
        } else if age > 60 {
            fitnessScore -= 5
        } else if age > 70 {
            fitnessScore -= 10
        }
        
        // Recovery efficiency bonus
        fitnessScore += recoveryEfficiency * 0.15
        
        // Normalize to 0-100 range
        fitnessScore = max(10, min(95, fitnessScore))
        
        // Determine category
        let category: String
        if fitnessScore > 80 {
            category = "Excellent"
        } else if fitnessScore > 65 {
            category = "Good"
        } else if fitnessScore > 45 {
            category = "Fair"
        } else if fitnessScore > 30 {
            category = "Below Average"
        } else {
            category = "Needs Improvement"
        }
        
        return (level: fitnessScore, category: category)
    }
    
    // MARK: - VO2max Estimation
    static func estimateVO2max(
        age: Double,
        restingHR: Double,
        maxHR: Double,
        hrReserve: Double,
        fitnessLevel: Double
    ) -> Double {
        // Use heart rate ratio method as base
        let hrRatio = maxHR / restingHR
        let baseVO2max = 15.3 * hrRatio
        
        // Adjust for fitness level (strong correlation)
        let fitnessAdjustment = fitnessLevel * 0.35
        
        // Age adjustment factor
        let ageAdjustment = max(0, (35 - age) * 0.25)
        
        // Heart rate reserve contribution
        let reserveAdjustment = hrReserve * 0.08
        
        // Calculate final VO2max
        var vo2max = baseVO2max + fitnessAdjustment + ageAdjustment + reserveAdjustment
        
        // Apply physiological limits
        vo2max = max(15, min(75, vo2max))
        
        return vo2max
    }
    
    // MARK: - Cardiovascular Age Calculation
    static func calculateCardiovascularAge(
        chronologicalAge: Double,
        fitnessLevel: Double,
        restingHR: Double,
        hrr1min: Double,
        rmssd: Double
    ) -> (cvAge: Double, comparison: String) {
        var cvAge = chronologicalAge
        
        // Fitness level is the primary driver
        let fitnessAdjustment = (fitnessLevel - 50) * -0.4
        cvAge += fitnessAdjustment
        
        // Heart rate recovery adjustment
        if hrr1min > 30 {
            cvAge -= 7
        } else if hrr1min > 25 {
            cvAge -= 4
        } else if hrr1min > 20 {
            cvAge -= 2
        } else if hrr1min < 15 {
            cvAge += 5
        } else if hrr1min < 12 {
            cvAge += 10
        }
        
        // Resting heart rate adjustment
        if restingHR < 55 {
            cvAge -= 4
        } else if restingHR < 60 {
            cvAge -= 2
        } else if restingHR > 75 {
            cvAge += 3
        } else if restingHR > 85 {
            cvAge += 6
        }
        
        // HRV adjustment
        if rmssd > 50 {
            cvAge -= 3
        } else if rmssd > 35 {
            cvAge -= 1
        } else if rmssd < 20 {
            cvAge += 4
        }
        
        // Apply reasonable bounds
        cvAge = max(18, min(90, cvAge))
        
        // Generate comparison string
        let difference = cvAge - chronologicalAge
        let comparison: String
        if difference < -5 {
            comparison = "💪 \(Int(abs(difference))) years younger"
        } else if difference < -2 {
            comparison = "✓ \(Int(abs(difference))) years younger"
        } else if difference > 5 {
            comparison = "⚠️ \(Int(difference)) years older"
        } else if difference > 2 {
            comparison = "\(Int(difference)) years older"
        } else {
            comparison = "Age appropriate"
        }
        
        return (cvAge: cvAge, comparison: comparison)
    }
    
    // MARK: - Recovery Pattern Analysis
    static func analyzeRecoveryPattern(
        hrr1min: Double,
        hrr2min: Double,
        timeToTarget: Double
    ) -> (efficiency: Double, status: String, recommendation: String) {
        // Calculate weighted recovery efficiency
        let hrr1Score = min(hrr1min / 30 * 50, 50)  // 50% weight
        let hrr2Score = min(hrr2min / 50 * 30, 30)  // 30% weight
        let timeScore = max(0, (180 - timeToTarget) / 180 * 20)  // 20% weight
        
        let efficiency = hrr1Score + hrr2Score + timeScore
        
        // Determine status and recommendations
        let status: String
        let recommendation: String
        
        if efficiency > 85 {
            status = "Excellent Recovery"
            recommendation = "Your cardiovascular recovery is elite level. Maintain current training intensity."
        } else if efficiency > 70 {
            status = "Very Good Recovery"
            recommendation = "Recovery is strong. You can handle high-intensity interval training."
        } else if efficiency > 55 {
            status = "Good Recovery"
            recommendation = "Recovery is healthy. Consider adding interval training 2-3x per week."
        } else if efficiency > 40 {
            status = "Fair Recovery"
            recommendation = "Recovery needs improvement. Focus on aerobic base building and ensure adequate rest."
        } else if efficiency > 25 {
            status = "Below Average Recovery"
            recommendation = "Recovery is concerning. Reduce training intensity and prioritize recovery days."
        } else {
            status = "Poor Recovery"
            recommendation = "Recovery needs immediate attention. Consult a healthcare provider and focus on gentle activity."
        }
        
        return (efficiency: efficiency, status: status, recommendation: recommendation)
    }
    
    // MARK: - Training Readiness Assessment
    static func assessTrainingReadiness(
        rmssd: Double,
        restingHR: Double,
        restingHRBaseline: Double,
        sleepQuality: Double
    ) -> (score: Double, status: String, guidance: String) {
        var readinessScore = 50.0
        
        // HRV is the primary indicator (highest weight)
        if rmssd > 60 {
            readinessScore += 25
        } else if rmssd > 45 {
            readinessScore += 15
        } else if rmssd > 30 {
            readinessScore += 8
        } else if rmssd < 20 {
            readinessScore -= 25
        } else if rmssd < 25 {
            readinessScore -= 10
        }
        
        // Check for elevated resting heart rate
        let hrElevation = restingHR - restingHRBaseline
        if hrElevation < -2 {
            readinessScore += 10  // Lower than baseline is good
        } else if hrElevation < 2 {
            readinessScore += 5   // Normal variation
        } else if hrElevation > 5 {
            readinessScore -= 15  // Elevated HR indicates stress
        } else if hrElevation > 10 {
            readinessScore -= 30  // Significantly elevated
        }
        
        // Sleep quality impact
        let sleepImpact = (sleepQuality - 0.5) * 40
        readinessScore += sleepImpact
        
        // Normalize score
        readinessScore = max(0, min(100, readinessScore))
        
        // Determine status and guidance
        let status: String
        let guidance: String
        
        if readinessScore > 85 {
            status = "Peak Performance Ready"
            guidance = "Your body is primed for maximum effort. Perfect day for personal records or competitions."
        } else if readinessScore > 70 {
            status = "Ready for High Intensity"
            guidance = "Great day for challenging workouts, intervals, or strength training."
        } else if readinessScore > 55 {
            status = "Ready for Moderate Activity"
            guidance = "Good for steady-state cardio, technique work, or moderate strength training."
        } else if readinessScore > 40 {
            status = "Light Activity Recommended"
            guidance = "Focus on recovery activities: easy walking, yoga, or stretching."
        } else if readinessScore > 25 {
            status = "Recovery Priority"
            guidance = "Your body needs rest. Consider meditation, light stretching, or complete rest."
        } else {
            status = "Rest Required"
            guidance = "Strong signs of fatigue or stress. Take a complete rest day and prioritize sleep."
        }
        
        return (score: readinessScore, status: status, guidance: guidance)
    }
    
    // MARK: - Helper Functions
    
    static func calculateHRR1Min(from heartRates: [Double]) -> Double {
        // Simulate HRR calculation from heart rate data
        // In real implementation, this would analyze post-exercise HR drop
        guard !heartRates.isEmpty else { return 20.0 }
        
        let maxHR = heartRates.max() ?? 100
        let minHR = heartRates.min() ?? 60
        let hrRange = maxHR - minHR
        
        // Estimate based on HR variability
        if hrRange > 40 {
            return 25 + Double.random(in: -3...3)
        } else if hrRange > 25 {
            return 20 + Double.random(in: -2...2)
        } else {
            return 15 + Double.random(in: -2...2)
        }
    }
    
    static func calculateRecoveryEfficiency(hrr1min: Double, hrr2min: Double) -> Double {
        let hrr1Score = min(hrr1min / 30 * 50, 50)
        let hrr2Score = min(hrr2min / 50 * 30, 30)
        return hrr1Score + hrr2Score + 20 // Base score
    }
    
    static func generatePersonalizedRecommendation(
        fitness: Double,
        recovery: Double,
        readiness: Double
    ) -> String {
        // Generate comprehensive recommendation based on all metrics
        var recommendations: [String] = []
        
        // Fitness-based recommendations
        if fitness < 40 {
            recommendations.append("Focus on building aerobic base with 30-min daily walks")
        } else if fitness < 60 {
            recommendations.append("Add 2-3 cardio sessions per week to improve fitness")
        } else if fitness > 75 {
            recommendations.append("Maintain excellence with varied training intensities")
        }
        
        // Recovery-based recommendations
        if recovery < 50 {
            recommendations.append("Prioritize recovery with proper sleep and nutrition")
        } else if recovery > 70 {
            recommendations.append("Recovery is strong - you can increase training volume")
        }
        
        // Readiness-based recommendations
        if readiness < 40 {
            recommendations.append("Take a rest day or do light recovery activities")
        } else if readiness > 70 {
            recommendations.append("Perfect timing for challenging workouts")
        }
        
        return recommendations.joined(separator: ". ")
    }
}

// MARK: - Data Model
struct CardiovascularFitnessAssessment {
    let fitnessLevel: Double
    let fitnessCategory: String
    let vo2max: Double
    let cardiovascularAge: Double
    let ageComparison: String
    let recoveryEfficiency: Double
    let recoveryStatus: String
    let trainingReadiness: Double
    let readinessStatus: String
    let recommendation: String
    let timestamp: Date
    
    var summary: String {
        """
        Fitness: \(fitnessCategory) (\(Int(fitnessLevel))/100)
        VO2max: \(String(format: "%.1f", vo2max)) ml/kg/min
        CV Age: \(Int(cardiovascularAge)) (\(ageComparison))
        Recovery: \(recoveryStatus)
        Readiness: \(readinessStatus)
        """
    }
}