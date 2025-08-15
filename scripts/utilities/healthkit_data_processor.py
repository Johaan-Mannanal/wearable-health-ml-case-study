#!/usr/bin/env python3
"""
HealthKit Data Processor for Heart Rhythm Classification

This script processes raw HealthKit data and prepares it for the improved SVM model.
It handles data extraction, cleaning, feature engineering, and model inference.

Compatible with:
- HKQuantityTypeIdentifierHeartRate
- HKQuantityTypeIdentifierHeartRateVariabilitySDNN
- HKQuantityTypeIdentifierHeartRateVariabilityRMSSD

Author: TelemetryHealthCare Team
Version: 2.0
"""

import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')


class HealthKitDataProcessor:
    """
    Process HealthKit data for heart rhythm classification.
    
    This class handles:
    1. Raw HealthKit data ingestion
    2. Feature extraction and engineering
    3. Data quality validation
    4. Model inference
    5. Result interpretation
    """
    
    def __init__(self, model_path: str = 'improved_heart_rhythm_svm_pipeline.pkl'):
        """
        Initialize the HealthKit data processor.
        
        Args:
            model_path: Path to the trained model pipeline
        """
        self.model_path = model_path
        self.model_pipeline = None
        self.metadata = None
        self.load_model()
    
    def load_model(self) -> None:
        """Load the trained model and metadata."""
        try:
            self.model_pipeline = joblib.load(self.model_path)
            print(f"✅ Model loaded successfully from {self.model_path}")
            
            # Load metadata if available
            try:
                with open('model_metadata.json', 'r') as f:
                    self.metadata = json.load(f)
                print(f"✅ Model metadata loaded (AUC: {self.metadata.get('auc_score', 'N/A'):.3f})")
            except FileNotFoundError:
                print("⚠️  Model metadata not found, using defaults")
                
        except FileNotFoundError:
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
    
    def process_healthkit_heart_rate(self, heart_rate_data: List[Dict]) -> pd.DataFrame:
        """
        Process HealthKit heart rate data.
        
        Expected format for heart_rate_data:
        [
            {
                'startDate': '2024-01-01T10:00:00',
                'endDate': '2024-01-01T10:01:00',
                'value': 72.0,
                'unit': 'count/min'
            },
            ...
        ]
        
        Args:
            heart_rate_data: List of heart rate measurements from HealthKit
            
        Returns:
            DataFrame with processed heart rate data
        """
        df = pd.DataFrame(heart_rate_data)
        
        if df.empty:
            raise ValueError("No heart rate data provided")
        
        # Convert timestamps
        df['startDate'] = pd.to_datetime(df['startDate'])
        df['endDate'] = pd.to_datetime(df['endDate'])
        
        # Validate heart rate values
        df = self._validate_heart_rate_data(df)
        
        # Sort by timestamp
        df = df.sort_values('startDate').reset_index(drop=True)
        
        print(f"✅ Processed {len(df)} heart rate measurements")
        return df
    
    def process_healthkit_hrv_data(self, hrv_data: List[Dict]) -> pd.DataFrame:
        """
        Process HealthKit HRV data (SDNN and RMSSD).
        
        Expected format for hrv_data:
        [
            {
                'startDate': '2024-01-01T10:00:00',
                'endDate': '2024-01-01T10:05:00',
                'value': 45.2,
                'unit': 'ms',
                'type': 'SDNN'  # or 'RMSSD'
            },
            ...
        ]
        
        Args:
            hrv_data: List of HRV measurements from HealthKit
            
        Returns:
            DataFrame with processed HRV data
        """
        df = pd.DataFrame(hrv_data)
        
        if df.empty:
            raise ValueError("No HRV data provided")
        
        # Convert timestamps
        df['startDate'] = pd.to_datetime(df['startDate'])
        df['endDate'] = pd.to_datetime(df['endDate'])
        
        # Validate HRV values
        df = self._validate_hrv_data(df)
        
        # Sort by timestamp
        df = df.sort_values('startDate').reset_index(drop=True)
        
        print(f"✅ Processed {len(df)} HRV measurements")
        return df
    
    def extract_features(self, 
                        heart_rate_df: pd.DataFrame, 
                        hrv_df: pd.DataFrame,
                        time_window_hours: int = 24) -> pd.DataFrame:
        """
        Extract features for heart rhythm classification.
        
        Args:
            heart_rate_df: Processed heart rate data
            hrv_df: Processed HRV data
            time_window_hours: Time window for feature calculation
            
        Returns:
            DataFrame with extracted features
        """
        features = []
        
        # Get the latest timestamp
        latest_time = max(
            heart_rate_df['startDate'].max(),
            hrv_df['startDate'].max()
        )
        
        # Define time window
        start_time = latest_time - timedelta(hours=time_window_hours)
        
        # Filter data to time window
        hr_window = heart_rate_df[
            (heart_rate_df['startDate'] >= start_time) &
            (heart_rate_df['startDate'] <= latest_time)
        ]
        
        hrv_window = hrv_df[
            (hrv_df['startDate'] >= start_time) &
            (hrv_df['startDate'] <= latest_time)
        ]
        
        if len(hr_window) < 10:  # Minimum data requirement
            raise ValueError(f"Insufficient heart rate data in {time_window_hours}h window: {len(hr_window)} samples")
        
        # Calculate mean heart rate
        mean_heart_rate = hr_window['value'].mean()
        
        # Calculate heart rate standard deviation
        std_heart_rate = hr_window['value'].std()
        
        # Calculate pNN50 from RMSSD data
        rmssd_data = hrv_window[hrv_window['type'] == 'RMSSD']
        if len(rmssd_data) > 0:
            # pNN50 estimation from RMSSD
            # This is a simplified calculation - in practice, you'd need RR intervals
            pnn50 = self._estimate_pnn50_from_rmssd(rmssd_data['value'].mean())
        else:
            # Fallback estimation from heart rate variability
            pnn50 = self._estimate_pnn50_from_hr_std(std_heart_rate)
        
        feature_row = {
            'timestamp': latest_time,
            'mean_heart_rate': mean_heart_rate,
            'std_heart_rate': std_heart_rate,
            'pnn50': pnn50,
            'data_quality_score': self._calculate_data_quality_score(hr_window, hrv_window),
            'sample_count_hr': len(hr_window),
            'sample_count_hrv': len(hrv_window)
        }
        
        features.append(feature_row)
        
        feature_df = pd.DataFrame(features)
        print(f"✅ Features extracted for {len(feature_df)} time windows")
        
        return feature_df
    
    def predict_rhythm(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict heart rhythm from extracted features.
        
        Args:
            features_df: DataFrame with extracted features
            
        Returns:
            DataFrame with predictions and confidence scores
        """
        if self.model_pipeline is None:
            raise ValueError("Model not loaded")
        
        # Prepare features for prediction
        feature_columns = ['mean_heart_rate', 'std_heart_rate', 'pnn50']
        X = features_df[feature_columns].values
        
        # Make predictions
        predictions = self.model_pipeline.predict(X)
        probabilities = self.model_pipeline.predict_proba(X)
        
        # Add results to dataframe
        results_df = features_df.copy()
        results_df['prediction'] = predictions
        results_df['normal_probability'] = probabilities[:, 0]
        results_df['irregular_probability'] = probabilities[:, 1]
        results_df['confidence'] = np.maximum(probabilities[:, 0], probabilities[:, 1])
        results_df['rhythm_classification'] = ['Normal' if p == 0 else 'Irregular' for p in predictions]
        
        return results_df
    
    def generate_health_report(self, results_df: pd.DataFrame) -> Dict:
        """
        Generate a comprehensive health report.
        
        Args:
            results_df: DataFrame with prediction results
            
        Returns:
            Dictionary containing health insights
        """
        latest_result = results_df.iloc[-1]
        
        report = {
            'timestamp': latest_result['timestamp'].isoformat(),
            'rhythm_classification': latest_result['rhythm_classification'],
            'confidence_score': float(latest_result['confidence']),
            'irregular_probability': float(latest_result['irregular_probability']),
            'heart_rate_metrics': {
                'mean_heart_rate': float(latest_result['mean_heart_rate']),
                'heart_rate_variability': float(latest_result['std_heart_rate']),
                'pnn50': float(latest_result['pnn50'])
            },
            'data_quality': {
                'quality_score': float(latest_result['data_quality_score']),
                'heart_rate_samples': int(latest_result['sample_count_hr']),
                'hrv_samples': int(latest_result['sample_count_hrv'])
            },
            'clinical_interpretation': self._generate_clinical_interpretation(latest_result),
            'recommendations': self._generate_recommendations(latest_result)
        }
        
        return report
    
    def _validate_heart_rate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean heart rate data."""
        # Remove invalid values
        original_count = len(df)
        df = df[(df['value'] >= 30) & (df['value'] <= 250)]  # Physiologically reasonable range
        
        if len(df) < original_count:
            print(f"⚠️  Removed {original_count - len(df)} invalid heart rate values")
        
        return df
    
    def _validate_hrv_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean HRV data."""
        original_count = len(df)
        df = df[(df['value'] >= 1) & (df['value'] <= 200)]  # Reasonable HRV range in ms
        
        if len(df) < original_count:
            print(f"⚠️  Removed {original_count - len(df)} invalid HRV values")
        
        return df
    
    def _estimate_pnn50_from_rmssd(self, rmssd_value: float) -> float:
        """
        Estimate pNN50 from RMSSD value.
        
        This is a simplified approximation. In practice, pNN50 should be
        calculated directly from RR intervals.
        """
        # Empirical relationship between RMSSD and pNN50
        # pNN50 ≈ (RMSSD / 150) ^ 1.5, capped at reasonable values
        estimated_pnn50 = min(0.6, max(0.0, (rmssd_value / 150) ** 1.5))
        return estimated_pnn50
    
    def _estimate_pnn50_from_hr_std(self, hr_std: float) -> float:
        """
        Estimate pNN50 from heart rate standard deviation.
        
        Fallback method when RMSSD is not available.
        """
        # Rough approximation: higher HR std suggests higher pNN50
        estimated_pnn50 = min(0.4, max(0.0, hr_std / 50))
        return estimated_pnn50
    
    def _calculate_data_quality_score(self, hr_df: pd.DataFrame, hrv_df: pd.DataFrame) -> float:
        """
        Calculate a data quality score (0-1).
        
        Higher scores indicate better data quality for reliable predictions.
        """
        score = 0.0
        
        # Heart rate data quality
        if len(hr_df) >= 50:  # Good sample size
            score += 0.4
        elif len(hr_df) >= 20:  # Acceptable sample size
            score += 0.2
        
        # HRV data availability
        if len(hrv_df) >= 5:
            score += 0.3
        elif len(hrv_df) >= 1:
            score += 0.1
        
        # Data consistency (low coefficient of variation indicates stable readings)
        if len(hr_df) > 1:
            cv = hr_df['value'].std() / hr_df['value'].mean()
            if cv < 0.3:  # Low variability suggests consistent readings
                score += 0.3
            elif cv < 0.5:
                score += 0.1
        
        return min(1.0, score)
    
    def _generate_clinical_interpretation(self, result: pd.Series) -> str:
        """Generate clinical interpretation of results."""
        rhythm = result['rhythm_classification']
        confidence = result['confidence']
        mean_hr = result['mean_heart_rate']
        
        if rhythm == 'Normal':
            if confidence > 0.8:
                return f"High confidence normal rhythm detected. Heart rate {mean_hr:.0f} BPM within normal range."
            else:
                return f"Likely normal rhythm, but consider additional monitoring. Heart rate {mean_hr:.0f} BPM."
        else:
            if confidence > 0.8:
                return f"High confidence irregular rhythm detected. Heart rate {mean_hr:.0f} BPM. Recommend medical evaluation."
            else:
                return f"Possible irregular rhythm detected. Heart rate {mean_hr:.0f} BPM. Consider further assessment."
    
    def _generate_recommendations(self, result: pd.Series) -> List[str]:
        """Generate personalized recommendations."""
        recommendations = []
        
        rhythm = result['rhythm_classification']
        confidence = result['confidence']
        data_quality = result['data_quality_score']
        
        if rhythm == 'Irregular':
            recommendations.append("Consult with a healthcare provider about irregular rhythm detection")
            recommendations.append("Continue regular monitoring with Apple Watch")
            if confidence < 0.7:
                recommendations.append("Consider additional ECG recordings for confirmation")
        
        if data_quality < 0.5:
            recommendations.append("Improve data quality by ensuring proper Apple Watch fit")
            recommendations.append("Take readings during rest periods for better accuracy")
        
        recommendations.append("Maintain regular physical activity as recommended by your doctor")
        recommendations.append("Continue monitoring trends over time")
        
        return recommendations


def process_sample_healthkit_data():
    """
    Demonstrate processing with sample HealthKit data.
    This function shows how to use the processor with realistic data.
    """
    print("🏥 Processing Sample HealthKit Data")
    print("=" * 50)
    
    # Create sample HealthKit heart rate data
    base_time = datetime.now() - timedelta(hours=2)
    heart_rate_data = []
    
    for i in range(120):  # 2 hours of data, every minute
        timestamp = base_time + timedelta(minutes=i)
        # Simulate some irregular periods
        if 30 <= i <= 45:  # Irregular period
            hr_value = np.random.normal(85, 15)
        else:  # Normal periods
            hr_value = np.random.normal(72, 8)
        
        heart_rate_data.append({
            'startDate': timestamp.isoformat(),
            'endDate': (timestamp + timedelta(minutes=1)).isoformat(),
            'value': max(50, min(150, hr_value)),
            'unit': 'count/min'
        })
    
    # Create sample HRV data
    hrv_data = []
    for i in range(24):  # Every 5 minutes
        timestamp = base_time + timedelta(minutes=i * 5)
        
        # Simulate different HRV patterns
        if 6 <= i <= 9:  # Period with lower HRV (might indicate stress/irregular rhythm)
            sdnn_value = np.random.normal(25, 5)
            rmssd_value = np.random.normal(20, 8)
        else:  # Normal HRV
            sdnn_value = np.random.normal(45, 10)
            rmssd_value = np.random.normal(35, 12)
        
        hrv_data.extend([
            {
                'startDate': timestamp.isoformat(),
                'endDate': (timestamp + timedelta(minutes=5)).isoformat(),
                'value': max(10, sdnn_value),
                'unit': 'ms',
                'type': 'SDNN'
            },
            {
                'startDate': timestamp.isoformat(),
                'endDate': (timestamp + timedelta(minutes=5)).isoformat(),
                'value': max(5, rmssd_value),
                'unit': 'ms',
                'type': 'RMSSD'
            }
        ])
    
    # Process the data
    try:
        processor = HealthKitDataProcessor()
        
        # Process raw data
        hr_df = processor.process_healthkit_heart_rate(heart_rate_data)
        hrv_df = processor.process_healthkit_hrv_data(hrv_data)
        
        # Extract features
        features_df = processor.extract_features(hr_df, hrv_df, time_window_hours=2)
        
        # Make predictions
        results_df = processor.predict_rhythm(features_df)
        
        # Generate report
        health_report = processor.generate_health_report(results_df)
        
        # Display results
        print("\n📊 Analysis Results:")
        print(f"Rhythm Classification: {health_report['rhythm_classification']}")
        print(f"Confidence: {health_report['confidence_score']:.1%}")
        print(f"Irregular Probability: {health_report['irregular_probability']:.1%}")
        
        print(f"\n💓 Heart Rate Metrics:")
        metrics = health_report['heart_rate_metrics']
        print(f"Mean HR: {metrics['mean_heart_rate']:.1f} BPM")
        print(f"HR Variability: {metrics['heart_rate_variability']:.1f} BPM")
        print(f"pNN50: {metrics['pnn50']:.3f}")
        
        print(f"\n📈 Data Quality:")
        quality = health_report['data_quality']
        print(f"Quality Score: {quality['quality_score']:.1%}")
        print(f"HR Samples: {quality['heart_rate_samples']}")
        print(f"HRV Samples: {quality['hrv_samples']}")
        
        print(f"\n🩺 Clinical Interpretation:")
        print(health_report['clinical_interpretation'])
        
        print(f"\n💡 Recommendations:")
        for rec in health_report['recommendations']:
            print(f"• {rec}")
        
        # Save sample results
        with open('sample_health_report.json', 'w') as f:
            json.dump(health_report, f, indent=2)
        
        print(f"\n✅ Sample report saved to 'sample_health_report.json'")
        
    except Exception as e:
        print(f"❌ Error processing sample data: {str(e)}")


if __name__ == "__main__":
    # Run sample processing
    process_sample_healthkit_data()
    
    print("\n" + "=" * 60)
    print("🍎 HealthKit Integration Guide")
    print("=" * 60)
    print("""
To integrate with your iOS app:

1. HealthKit Data Collection:
   - Request permission for HKQuantityTypeIdentifierHeartRate
   - Request permission for HKQuantityTypeIdentifierHeartRateVariabilitySDNN
   - Request permission for HKQuantityTypeIdentifierHeartRateVariabilityRMSSD

2. Data Export:
   - Export heart rate data as JSON with timestamps
   - Export HRV data with SDNN and RMSSD values
   - Ensure data covers at least 1-2 hours for reliable analysis

3. Python Integration:
   - Use this script to process exported HealthKit data
   - Call predict_rhythm() for real-time analysis
   - Implement health_report generation for user insights

4. Model Updates:
   - Retrain model periodically with real patient data
   - Validate predictions against medical professionals
   - Consider regulatory compliance for medical applications

Remember: This model is for research purposes and should not
replace professional medical diagnosis.
    """)