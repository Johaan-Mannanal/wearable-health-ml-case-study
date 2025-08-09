#!/usr/bin/env python3
"""
Cardiovascular Fitness & Recovery Model Training
================================================
This model analyzes medium/long-term cardiovascular health by examining:
1. Heart Rate Recovery (HRR) patterns
2. Resting Heart Rate trends
3. Exercise response efficiency
4. Autonomic nervous system balance
5. Cardiovascular age estimation
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.pipeline import Pipeline
import xgboost as xgb
import joblib
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

class CardiovascularFitnessDataGenerator:
    """Generate realistic cardiovascular fitness training data"""
    
    def __init__(self, n_samples=10000):
        self.n_samples = n_samples
        
    def generate_fitness_profile(self):
        """Generate a single user's fitness profile"""
        # Age affects cardiovascular parameters
        age = np.random.randint(18, 80)
        
        # Fitness level (0-100 scale)
        # Younger people tend to have better baseline fitness
        base_fitness = 70 - (age - 40) * 0.5 + np.random.normal(0, 10)
        base_fitness = np.clip(base_fitness, 10, 95)
        
        # Activity level affects fitness
        activity_level = np.random.choice(['sedentary', 'light', 'moderate', 'active', 'athlete'], 
                                        p=[0.25, 0.30, 0.25, 0.15, 0.05])
        
        activity_multiplier = {
            'sedentary': 0.7,
            'light': 0.85,
            'moderate': 1.0,
            'active': 1.15,
            'athlete': 1.3
        }[activity_level]
        
        adjusted_fitness = base_fitness * activity_multiplier
        adjusted_fitness = np.clip(adjusted_fitness, 10, 95)
        
        return age, adjusted_fitness, activity_level
    
    def generate_cardiovascular_metrics(self, age, fitness_level, activity_level):
        """Generate realistic cardiovascular metrics based on fitness profile"""
        
        # Resting Heart Rate (lower is generally better)
        # Elite athletes: 40-50, Good fitness: 50-60, Average: 60-70, Poor: 70-85
        base_rhr = 85 - fitness_level * 0.35
        rhr = base_rhr + np.random.normal(0, 3)
        rhr = np.clip(rhr, 40, 95)
        
        # Maximum Heart Rate (age-predicted)
        max_hr = 220 - age + np.random.normal(0, 5)
        
        # Heart Rate Reserve (max HR - resting HR)
        hr_reserve = max_hr - rhr
        
        # Heart Rate Recovery - 1 minute post-exercise
        # Excellent: >30 bpm, Good: 20-30, Fair: 12-20, Poor: <12
        if fitness_level > 70:
            hrr_1min = 30 + (fitness_level - 70) * 0.5 + np.random.normal(0, 3)
        elif fitness_level > 40:
            hrr_1min = 20 + (fitness_level - 40) * 0.33 + np.random.normal(0, 2)
        else:
            hrr_1min = 12 + (fitness_level - 20) * 0.4 + np.random.normal(0, 2)
        hrr_1min = np.clip(hrr_1min, 5, 50)
        
        # Heart Rate Recovery - 2 minutes post-exercise
        hrr_2min = hrr_1min * 1.5 + np.random.normal(0, 3)
        hrr_2min = np.clip(hrr_2min, 10, 70)
        
        # Exercise Heart Rate at moderate intensity (60-70% of max)
        exercise_hr_moderate = rhr + (hr_reserve * 0.65) + np.random.normal(0, 5)
        
        # Time to reach target heart rate during exercise (seconds)
        # Fitter individuals reach target HR more efficiently
        time_to_target_hr = 180 - fitness_level * 1.5 + np.random.normal(0, 10)
        time_to_target_hr = np.clip(time_to_target_hr, 30, 240)
        
        # HRV metrics (higher is generally better for fitness)
        # RMSSD - Root Mean Square of Successive Differences
        rmssd = 20 + fitness_level * 0.6 + np.random.normal(0, 5)
        rmssd = np.clip(rmssd, 10, 100)
        
        # SDNN - Standard Deviation of NN intervals
        sdnn = 30 + fitness_level * 0.7 + np.random.normal(0, 8)
        sdnn = np.clip(sdnn, 20, 120)
        
        # pNN50 - percentage of successive RR intervals that differ by more than 50ms
        pnn50 = fitness_level * 0.3 + np.random.normal(0, 3)
        pnn50 = np.clip(pnn50, 0, 40)
        
        # Autonomic Balance Score (LF/HF ratio)
        # Lower values (0.5-1.5) indicate better parasympathetic tone
        if fitness_level > 60:
            lf_hf_ratio = 0.8 + np.random.normal(0, 0.2)
        else:
            lf_hf_ratio = 1.5 + (60 - fitness_level) * 0.02 + np.random.normal(0, 0.3)
        lf_hf_ratio = np.clip(lf_hf_ratio, 0.3, 4.0)
        
        # Recovery efficiency score (0-100)
        recovery_efficiency = (hrr_1min / 50) * 40 + (hrr_2min / 70) * 30 + (100 - time_to_target_hr/2.4) * 0.3
        recovery_efficiency = np.clip(recovery_efficiency, 0, 100)
        
        # Training load tolerance (how well they handle exercise stress)
        training_tolerance = fitness_level * 0.8 + rmssd * 0.2 + np.random.normal(0, 5)
        training_tolerance = np.clip(training_tolerance, 10, 100)
        
        # Circadian consistency score (0-100)
        # How consistent are their daily HR patterns
        circadian_consistency = 70 + fitness_level * 0.2 + np.random.normal(0, 10)
        circadian_consistency = np.clip(circadian_consistency, 30, 95)
        
        # VO2max estimate (ml/kg/min) - gold standard for cardiovascular fitness
        # Based on heart rate reserve and other factors
        if activity_level == 'athlete':
            vo2max = 50 + fitness_level * 0.3 + np.random.normal(0, 3)
        elif activity_level == 'active':
            vo2max = 40 + fitness_level * 0.25 + np.random.normal(0, 3)
        elif activity_level == 'moderate':
            vo2max = 35 + fitness_level * 0.2 + np.random.normal(0, 3)
        else:
            vo2max = 25 + fitness_level * 0.2 + np.random.normal(0, 3)
        vo2max = np.clip(vo2max, 15, 75)
        
        # Cardiovascular age (biological age of cardiovascular system)
        # Can be younger or older than chronological age based on fitness
        fitness_age_adjustment = (fitness_level - 50) * -0.3  # Higher fitness = younger CV age
        cardiovascular_age = age + fitness_age_adjustment + np.random.normal(0, 3)
        cardiovascular_age = np.clip(cardiovascular_age, 18, 90)
        
        return {
            'age': age,
            'resting_hr': rhr,
            'max_hr': max_hr,
            'hr_reserve': hr_reserve,
            'hrr_1min': hrr_1min,
            'hrr_2min': hrr_2min,
            'exercise_hr_moderate': exercise_hr_moderate,
            'time_to_target_hr': time_to_target_hr,
            'rmssd': rmssd,
            'sdnn': sdnn,
            'pnn50': pnn50,
            'lf_hf_ratio': lf_hf_ratio,
            'recovery_efficiency': recovery_efficiency,
            'training_tolerance': training_tolerance,
            'circadian_consistency': circadian_consistency,
            'vo2max': vo2max,
            'cardiovascular_age': cardiovascular_age,
            'fitness_level': fitness_level,
            'activity_level': activity_level
        }
    
    def add_temporal_features(self, data):
        """Add temporal trend features for long-term monitoring"""
        # Simulate trends over time
        data['rhr_trend_30d'] = np.random.normal(0, 2)  # Change in RHR over 30 days
        data['hrv_trend_30d'] = np.random.normal(0, 5)  # Change in HRV over 30 days
        data['fitness_trend_90d'] = np.random.normal(0, 5)  # Change in fitness over 90 days
        
        # Weekly variability
        data['rhr_weekly_std'] = np.random.uniform(1, 5)  # Consistency of RHR
        data['hrv_weekly_std'] = np.random.uniform(3, 15)  # Consistency of HRV
        
        return data
    
    def generate_dataset(self):
        """Generate complete dataset"""
        data = []
        
        for _ in range(self.n_samples):
            age, fitness_level, activity_level = self.generate_fitness_profile()
            metrics = self.generate_cardiovascular_metrics(age, fitness_level, activity_level)
            metrics = self.add_temporal_features(metrics)
            data.append(metrics)
        
        return pd.DataFrame(data)

class CardiovascularFitnessModel:
    """Train and evaluate cardiovascular fitness models"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        
    def prepare_features(self, df):
        """Prepare features for training"""
        # Define feature groups
        self.feature_cols = [
            'age', 'resting_hr', 'hr_reserve', 'hrr_1min', 'hrr_2min',
            'exercise_hr_moderate', 'time_to_target_hr', 'rmssd', 'sdnn', 
            'pnn50', 'lf_hf_ratio', 'recovery_efficiency', 'training_tolerance',
            'circadian_consistency', 'rhr_trend_30d', 'hrv_trend_30d', 
            'fitness_trend_90d', 'rhr_weekly_std', 'hrv_weekly_std'
        ]
        
        # Target variables
        self.targets = {
            'fitness_level': df['fitness_level'].values,
            'vo2max': df['vo2max'].values,
            'cardiovascular_age': df['cardiovascular_age'].values
        }
        
        # Create feature matrix
        X = df[self.feature_cols].values
        
        return X
    
    def train_fitness_level_model(self, X, y):
        """Train model to predict overall fitness level"""
        print("\n" + "="*60)
        print("Training Fitness Level Prediction Model")
        print("="*60)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Create models
        models = [
            ('rf', RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42)),
            ('gb', GradientBoostingRegressor(n_estimators=150, max_depth=10, random_state=42)),
            ('xgb', xgb.XGBRegressor(n_estimators=200, max_depth=12, learning_rate=0.1, random_state=42))
        ]
        
        # Create ensemble
        ensemble = VotingRegressor(models)
        
        # Create pipeline with scaling
        pipeline = Pipeline([
            ('scaler', RobustScaler()),
            ('model', ensemble)
        ])
        
        # Train
        pipeline.fit(X_train, y_train)
        
        # Evaluate
        y_pred = pipeline.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        print(f"Fitness Level Model Performance:")
        print(f"  MAE: {mae:.2f}")
        print(f"  RMSE: {rmse:.2f}")
        print(f"  R²: {r2:.3f}")
        
        # Cross-validation
        cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='r2')
        print(f"  Cross-validation R² (mean ± std): {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        
        self.models['fitness_level'] = pipeline
        
        return pipeline
    
    def train_vo2max_model(self, X, y):
        """Train model to predict VO2max"""
        print("\n" + "="*60)
        print("Training VO2max Prediction Model")
        print("="*60)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Neural network for VO2max prediction
        nn_model = MLPRegressor(
            hidden_layer_sizes=(100, 50, 25),
            activation='relu',
            solver='adam',
            max_iter=500,
            random_state=42
        )
        
        # Create pipeline
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', nn_model)
        ])
        
        # Train
        pipeline.fit(X_train, y_train)
        
        # Evaluate
        y_pred = pipeline.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        print(f"VO2max Model Performance:")
        print(f"  MAE: {mae:.2f} ml/kg/min")
        print(f"  RMSE: {rmse:.2f} ml/kg/min")
        print(f"  R²: {r2:.3f}")
        
        # Cross-validation
        cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='r2')
        print(f"  Cross-validation R² (mean ± std): {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        
        self.models['vo2max'] = pipeline
        
        return pipeline
    
    def train_cardiovascular_age_model(self, X, y):
        """Train model to predict cardiovascular age"""
        print("\n" + "="*60)
        print("Training Cardiovascular Age Model")
        print("="*60)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # XGBoost for cardiovascular age
        xgb_model = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=10,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        
        # Create pipeline
        pipeline = Pipeline([
            ('scaler', RobustScaler()),
            ('model', xgb_model)
        ])
        
        # Train
        pipeline.fit(X_train, y_train)
        
        # Evaluate
        y_pred = pipeline.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        print(f"Cardiovascular Age Model Performance:")
        print(f"  MAE: {mae:.1f} years")
        print(f"  RMSE: {rmse:.1f} years")
        print(f"  R²: {r2:.3f}")
        
        # Cross-validation
        cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='r2')
        print(f"  Cross-validation R² (mean ± std): {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        
        self.models['cardiovascular_age'] = pipeline
        
        # Extract feature importance
        if hasattr(xgb_model, 'feature_importances_'):
            importance = xgb_model.feature_importances_
            feature_importance = dict(zip(self.feature_cols, importance))
            sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            
            print("\nTop 10 Most Important Features for Cardiovascular Age:")
            for feature, imp in sorted_importance[:10]:
                print(f"  {feature}: {imp:.3f}")
        
        return pipeline
    
    def validate_models(self, df):
        """Comprehensive model validation"""
        print("\n" + "="*60)
        print("Model Validation on Different Age Groups")
        print("="*60)
        
        X = self.prepare_features(df)
        
        # Test on different age groups
        age_groups = [
            (18, 30, "Young Adults"),
            (31, 50, "Middle Age"),
            (51, 70, "Older Adults"),
            (71, 80, "Elderly")
        ]
        
        for min_age, max_age, group_name in age_groups:
            mask = (df['age'] >= min_age) & (df['age'] <= max_age)
            X_group = X[mask]
            
            if len(X_group) > 0:
                print(f"\n{group_name} ({min_age}-{max_age} years):")
                
                # Fitness level prediction
                y_true = df.loc[mask, 'fitness_level'].values
                y_pred = self.models['fitness_level'].predict(X_group)
                mae = mean_absolute_error(y_true, y_pred)
                print(f"  Fitness Level MAE: {mae:.2f}")
                
                # VO2max prediction
                y_true = df.loc[mask, 'vo2max'].values
                y_pred = self.models['vo2max'].predict(X_group)
                mae = mean_absolute_error(y_true, y_pred)
                print(f"  VO2max MAE: {mae:.2f} ml/kg/min")
                
                # Cardiovascular age prediction
                y_true = df.loc[mask, 'cardiovascular_age'].values
                y_pred = self.models['cardiovascular_age'].predict(X_group)
                mae = mean_absolute_error(y_true, y_pred)
                print(f"  CV Age MAE: {mae:.1f} years")
    
    def save_models(self):
        """Save trained models"""
        print("\n" + "="*60)
        print("Saving Models")
        print("="*60)
        
        # Save each model
        for name, model in self.models.items():
            filename = f'cardiovascular_{name}_model.pkl'
            joblib.dump(model, filename)
            print(f"Saved {name} model to {filename}")
        
        # Save feature configuration
        config = {
            'feature_cols': self.feature_cols,
            'model_names': list(self.models.keys()),
            'timestamp': datetime.now().isoformat()
        }
        
        with open('cardiovascular_model_config.json', 'w') as f:
            json.dump(config, f, indent=2)
        print("Saved model configuration")
    
    def generate_swift_implementation(self):
        """Generate Swift code for iOS implementation"""
        print("\n" + "="*60)
        print("Generating Swift Implementation")
        print("="*60)
        
        swift_code = '''
// Cardiovascular Fitness & Recovery Model
// Auto-generated from Python training script

import Foundation

class CardiovascularFitnessModel {
    
    // MARK: - Fitness Level Prediction
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
        // Simplified implementation based on model patterns
        var fitnessScore = 50.0
        
        // Heart rate recovery is the strongest predictor
        if hrr1min > 30 {
            fitnessScore += 20
        } else if hrr1min > 20 {
            fitnessScore += 10
        } else if hrr1min < 12 {
            fitnessScore -= 15
        }
        
        // Resting heart rate contribution
        if restingHR < 55 {
            fitnessScore += 15
        } else if restingHR < 65 {
            fitnessScore += 8
        } else if restingHR > 75 {
            fitnessScore -= 10
        }
        
        // HRV contribution
        if rmssd > 50 {
            fitnessScore += 10
        } else if rmssd < 20 {
            fitnessScore -= 10
        }
        
        // Age adjustment
        let ageAdjustment = max(0, (40 - age) * 0.3)
        fitnessScore += ageAdjustment
        
        // Recovery efficiency
        fitnessScore += recoveryEfficiency * 0.2
        
        // Clamp to valid range
        fitnessScore = max(10, min(95, fitnessScore))
        
        // Categorize
        let category: String
        if fitnessScore > 80 {
            category = "Excellent"
        } else if fitnessScore > 65 {
            category = "Good"
        } else if fitnessScore > 45 {
            category = "Fair"
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
        // Simplified VO2max estimation
        // Based on heart rate reserve method
        let baseVO2max = 15.3 * (maxHR / restingHR)
        
        // Adjust for fitness level
        let fitnessAdjustment = fitnessLevel * 0.25
        
        // Age adjustment
        let ageAdjustment = max(0, (30 - age) * 0.2)
        
        var vo2max = baseVO2max + fitnessAdjustment + ageAdjustment
        
        // Clamp to physiological range
        vo2max = max(15, min(75, vo2max))
        
        return vo2max
    }
    
    // MARK: - Cardiovascular Age
    static func calculateCardiovascularAge(
        chronologicalAge: Double,
        fitnessLevel: Double,
        restingHR: Double,
        hrr1min: Double,
        rmssd: Double
    ) -> (cvAge: Double, comparison: String) {
        var cvAge = chronologicalAge
        
        // Fitness level adjustment (most important)
        let fitnessAdjustment = (fitnessLevel - 50) * -0.3
        cvAge += fitnessAdjustment
        
        // Heart rate recovery adjustment
        if hrr1min > 25 {
            cvAge -= 5
        } else if hrr1min < 15 {
            cvAge += 8
        }
        
        // Resting HR adjustment
        if restingHR < 60 {
            cvAge -= 3
        } else if restingHR > 75 {
            cvAge += 5
        }
        
        // HRV adjustment
        if rmssd > 40 {
            cvAge -= 2
        } else if rmssd < 20 {
            cvAge += 3
        }
        
        // Clamp to reasonable range
        cvAge = max(18, min(90, cvAge))
        
        // Generate comparison
        let difference = cvAge - chronologicalAge
        let comparison: String
        if difference < -5 {
            comparison = "\\(Int(abs(difference))) years younger"
        } else if difference > 5 {
            comparison = "\\(Int(difference)) years older"
        } else {
            comparison = "Age appropriate"
        }
        
        return (cvAge: cvAge, comparison: comparison)
    }
    
    // MARK: - Recovery Analysis
    static func analyzeRecoveryPattern(
        hrr1min: Double,
        hrr2min: Double,
        timeToTarget: Double
    ) -> (efficiency: Double, status: String, recommendation: String) {
        // Calculate recovery efficiency score
        let hrr1Score = min(hrr1min / 30 * 50, 50)  // 50% weight
        let hrr2Score = min(hrr2min / 50 * 30, 30)  // 30% weight
        let timeScore = max(0, (180 - timeToTarget) / 180 * 20)  // 20% weight
        
        let efficiency = hrr1Score + hrr2Score + timeScore
        
        // Determine status
        let status: String
        let recommendation: String
        
        if efficiency > 80 {
            status = "Excellent Recovery"
            recommendation = "Your cardiovascular recovery is optimal. Maintain current training."
        } else if efficiency > 60 {
            status = "Good Recovery"
            recommendation = "Recovery is healthy. Consider interval training to improve further."
        } else if efficiency > 40 {
            status = "Fair Recovery"
            recommendation = "Recovery could improve. Add more cardio and ensure adequate rest."
        } else {
            status = "Poor Recovery"
            recommendation = "Recovery needs attention. Consult healthcare provider and focus on gradual conditioning."
        }
        
        return (efficiency: efficiency, status: status, recommendation: recommendation)
    }
    
    // MARK: - Training Readiness
    static func assessTrainingReadiness(
        rmssd: Double,
        restingHR: Double,
        restingHRBaseline: Double,
        sleepQuality: Double
    ) -> (score: Double, status: String, guidance: String) {
        var readinessScore = 50.0
        
        // HRV contribution (most important for readiness)
        if rmssd > 50 {
            readinessScore += 20
        } else if rmssd > 30 {
            readinessScore += 10
        } else if rmssd < 20 {
            readinessScore -= 20
        }
        
        // Resting HR elevation from baseline
        let hrElevation = restingHR - restingHRBaseline
        if hrElevation < 0 {
            readinessScore += 10
        } else if hrElevation > 5 {
            readinessScore -= 15
        } else if hrElevation > 10 {
            readinessScore -= 25
        }
        
        // Sleep quality impact
        readinessScore += sleepQuality * 20
        
        // Clamp score
        readinessScore = max(0, min(100, readinessScore))
        
        // Determine status and guidance
        let status: String
        let guidance: String
        
        if readinessScore > 80 {
            status = "Ready for High Intensity"
            guidance = "Your body is well-recovered. Today is ideal for challenging workouts."
        } else if readinessScore > 60 {
            status = "Ready for Moderate Activity"
            guidance = "Good for steady-state cardio or moderate training."
        } else if readinessScore > 40 {
            status = "Light Activity Recommended"
            guidance = "Focus on recovery activities like walking or yoga."
        } else {
            status = "Rest Recommended"
            guidance = "Your body needs recovery. Prioritize rest and sleep."
        }
        
        return (score: readinessScore, status: status, guidance: guidance)
    }
}
'''
        
        with open('CardiovascularFitnessModel.swift', 'w') as f:
            f.write(swift_code)
        
        print("Generated Swift implementation: CardiovascularFitnessModel.swift")

def main():
    """Main training pipeline"""
    print("="*60)
    print("CARDIOVASCULAR FITNESS MODEL TRAINING")
    print("="*60)
    
    # Generate training data
    print("\nGenerating training data...")
    generator = CardiovascularFitnessDataGenerator(n_samples=10000)
    df = generator.generate_dataset()
    
    print(f"Generated {len(df)} samples")
    print(f"Features: {df.shape[1]}")
    
    # Display sample statistics
    print("\nDataset Statistics:")
    print(df[['age', 'resting_hr', 'hrr_1min', 'vo2max', 'fitness_level', 'cardiovascular_age']].describe())
    
    # Initialize model trainer
    trainer = CardiovascularFitnessModel()
    
    # Prepare features
    X = trainer.prepare_features(df)
    
    # Train models
    trainer.train_fitness_level_model(X, trainer.targets['fitness_level'])
    trainer.train_vo2max_model(X, trainer.targets['vo2max'])
    trainer.train_cardiovascular_age_model(X, trainer.targets['cardiovascular_age'])
    
    # Validate models
    trainer.validate_models(df)
    
    # Save models
    trainer.save_models()
    
    # Generate Swift implementation
    trainer.generate_swift_implementation()
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print("\nModels saved:")
    print("  - cardiovascular_fitness_level_model.pkl")
    print("  - cardiovascular_vo2max_model.pkl")
    print("  - cardiovascular_cardiovascular_age_model.pkl")
    print("  - cardiovascular_model_config.json")
    print("  - CardiovascularFitnessModel.swift")
    
    print("\nNext steps:")
    print("1. Review the Swift implementation")
    print("2. Integrate into the iOS app")
    print("3. Test with real Apple Watch data")
    print("4. Fine-tune thresholds based on user feedback")

if __name__ == "__main__":
    main()