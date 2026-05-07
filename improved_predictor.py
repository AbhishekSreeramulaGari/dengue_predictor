"""
Improved Dengue Prediction Model
Provides realistic risk assessments based on complaint data and environmental factors
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
from datetime import datetime
import joblib
import warnings

warnings.filterwarnings('ignore')


class ImprovedDenguePredictor:
    """Improved dengue prediction with realistic risk assessment"""

    def __init__(self):
        """Initialize the improved predictor"""
        self.model = None
        self.scaler = None
        self.load_model()

        # Risk thresholds based on realistic dengue case numbers
        self.risk_thresholds = {
            'Low': (0, 15),
            'Medium': (15, 45),
            'High': (45, 100),
            'Very High': (100, float('inf'))
        }

        # Environmental factors that influence dengue risk
        self.environmental_factors = {
            'rainfall': {'low': 50, 'medium': 150, 'high': 250},
            'temperature': {'low': 20, 'optimal': 28, 'high': 35},
            'humidity': {'low': 40, 'optimal': 70, 'high': 90}
        }

    def load_model(self):
        """Load the trained model and scaler"""
        try:
            self.model = joblib.load('model_xgboost_final.joblib')
            self.scaler = joblib.load('scaler_global_final.joblib')
        except Exception:
            try:
                self.model = joblib.load('model_xgboost.joblib')
                self.scaler = joblib.load('scaler.joblib')
            except Exception:
                self.model = None
                self.scaler = None

    def calculate_complaint_severity_score(self, garbage_count: float, waterlogging_count: float,
                                         garbage_text: str = "", waterlogging_text: str = "") -> float:
        """
        Calculate severity score based on complaint counts and descriptions

        Args:
            garbage_count: Number of garbage complaints
            waterlogging_count: Number of waterlogging complaints
            garbage_text: Description of garbage issues
            waterlogging_text: Description of waterlogging issues

        Returns:
            Severity score (0-10)
        """

        # Base score from complaint counts
        base_score = min(10, (garbage_count + waterlogging_count) * 0.5)

        # Text analysis for severity multipliers
        severity_multiplier = 1.0

        # Keywords that increase severity
        high_severity_keywords = [
            'overflowing', 'stagnant', 'breeding', 'mosquito', 'epidemic',
            'continuous', 'persistent', 'severe', 'critical', 'emergency'
        ]

        # Keywords that moderately increase severity
        medium_severity_keywords = [
            'clogged', 'blocked', 'accumulation', 'poor', 'inadequate',
            'insufficient', 'frequent', 'regular', 'ongoing'
        ]

        all_text = f"{garbage_text} {waterlogging_text}".lower()

        high_count = sum(1 for keyword in high_severity_keywords if keyword in all_text)
        medium_count = sum(1 for keyword in medium_severity_keywords if keyword in all_text)

        severity_multiplier += high_count * 0.3 + medium_count * 0.15
        severity_multiplier = min(2.5, severity_multiplier)

        # Calculate final severity score
        severity_score = min(10.0, base_score * severity_multiplier)

        return severity_score

    def get_environmental_risk_factor(self, month: int = None) -> float:
        """
        Calculate environmental risk factor based on season/month

        Args:
            month: Month number (1-12), defaults to current month

        Returns:
            Environmental risk factor (0.5-2.0)
        """

        if month is None:
            month = datetime.now().month

        # Monsoon season (June-September) has highest risk
        if month in [6, 7, 8, 9]:
            return 1.8
        # Pre/post monsoon (May, October)
        elif month in [5, 10]:
            return 1.4
        # Dry season (November-April)
        else:
            return 0.7

    def prepare_prediction_features(self, garbage_count: float, waterlogging_count: float,
                                  garbage_text: str = "", waterlogging_text: str = "",
                                  ward_id: int = None) -> pd.DataFrame:
        """
        Prepare features for prediction model

        Args:
            garbage_count: Number of garbage complaints
            waterlogging_count: Number of waterlogging complaints
            garbage_text: Description of garbage issues
            waterlogging_text: Description of waterlogging issues
            ward_id: Ward identifier

        Returns:
            DataFrame with prediction features
        """

        # Calculate severity scores
        garbage_severity = self.calculate_complaint_severity_score(garbage_count, 0, garbage_text, "")
        waterlogging_severity = self.calculate_complaint_severity_score(0, waterlogging_count, "", waterlogging_text)

        # Environmental factors (using realistic values)
        current_month = datetime.now().month
        environmental_factor = self.get_environmental_risk_factor(current_month)

        rainfall = 120 * environmental_factor  # Base rainfall adjusted by season
        temperature = 28 + (current_month - 6) * 0.5  # Temperature variation
        temperature = max(20, min(35, temperature))

        # Historical data (lagged values)
        rain_lag1 = rainfall * 0.8
        rain_lag2 = rainfall * 0.6
        temp_lag1 = temperature - 1
        cases_lag1 = max(5, (garbage_severity + waterlogging_severity) * 2)

        # Rolling averages
        rainfall_roll3_mean = (rainfall + rain_lag1 + rain_lag2) / 3
        cases_roll3_mean = max(5, cases_lag1 * 0.8)

        # Monsoon indicator
        is_monsoon = 1 if rainfall > 180 else 0

        # Ward-level complaint means (simplified)
        garbage_ward_mean = garbage_severity * 0.8
        waterlogging_ward_mean = waterlogging_severity * 0.8

        features = {
            'Rainfall_mm': rainfall,
            'Avg_Temp_C': temperature,
            'Garbage_Complaints': garbage_severity,
            'Waterlogging_Complaints': waterlogging_severity,
            'Rainfall_Lag1': rain_lag1,
            'Rainfall_Lag2': rain_lag2,
            'Temp_Lag1': temp_lag1,
            'Cases_Lag1': cases_lag1,
            'Rainfall_roll3_mean': rainfall_roll3_mean,
            'Cases_roll3_mean': cases_roll3_mean,
            'Is_Monsoon': is_monsoon,
            'Garbage_Complaints_ward_mean': garbage_ward_mean,
            'Waterlogging_Complaints_ward_mean': waterlogging_ward_mean
        }

        return pd.DataFrame([features])

    def predict_dengue_cases(self, features_df: pd.DataFrame) -> float:
        """
        Predict dengue cases using the trained model

        Args:
            features_df: DataFrame with prediction features

        Returns:
            Predicted number of dengue cases
        """

        if self.model is None:
            # Fallback prediction based on complaint severity
            garbage_severity = features_df['Garbage_Complaints'].iloc[0]
            waterlogging_severity = features_df['Waterlogging_Complaints'].iloc[0]
            environmental_factor = features_df['Rainfall_mm'].iloc[0] / 120

            base_prediction = (garbage_severity + waterlogging_severity) * environmental_factor * 2
            return max(0, min(150, base_prediction))

        try:
            if self.scaler:
                features_scaled = self.scaler.transform(features_df)
            else:
                features_scaled = features_df.values

            prediction = self.model.predict(features_scaled)[0]

            # Apply realistic bounds and calibration
            prediction = max(0, float(prediction))

            # Calibrate based on environmental factors
            environmental_factor = features_df['Rainfall_mm'].iloc[0] / 120
            prediction *= environmental_factor

            # Apply reasonable upper bound
            prediction = min(200, prediction)

            return prediction

        except Exception as e:
            print(f"Prediction error: {e}")
            # Fallback prediction
            garbage_severity = features_df['Garbage_Complaints'].iloc[0]
            waterlogging_severity = features_df['Waterlogging_Complaints'].iloc[0]
            return max(0, (garbage_severity + waterlogging_severity) * 1.5)

    def classify_risk_level(self, predicted_cases: float) -> Tuple[str, str, str]:
        """
        Classify risk level based on predicted cases

        Args:
            predicted_cases: Predicted number of dengue cases

        Returns:
            Tuple of (risk_level, bootstrap_color, description)
        """

        if predicted_cases <= 10:
            return 'Low Risk', 'success', 'Minimal dengue activity expected. Maintain regular preventive measures.'
        elif predicted_cases <= 30:
            return 'Medium Risk', 'warning', 'Moderate dengue risk. Increase mosquito control and eliminate breeding sites.'
        elif predicted_cases <= 70:
            return 'High Risk', 'danger', 'High dengue risk. Implement intensive vector control and community awareness programs.'
        else:
            return 'Very High Risk', 'dark', 'Critical dengue risk. Immediate intervention required with medical surveillance.'

    def get_prevention_recommendations(self, risk_level: str) -> List[str]:
        """
        Get prevention recommendations based on risk level

        Args:
            risk_level: Risk level string

        Returns:
            List of prevention recommendations
        """

        base_recommendations = [
            "Eliminate standing water in containers and gutters",
            "Use mosquito repellents and wear protective clothing",
            "Install window and door screens",
            "Keep surroundings clean and free of garbage"
        ]

        if risk_level == 'Low Risk':
            return base_recommendations
        elif risk_level == 'Medium Risk':
            return base_recommendations + [
                "Conduct weekly fogging in high-risk areas",
                "Increase community awareness campaigns",
                "Monitor for mosquito breeding sites more frequently"
            ]
        elif risk_level == 'High Risk':
            return base_recommendations + [
                "Implement daily vector control activities",
                "Set up medical surveillance systems",
                "Distribute larvicides in affected areas",
                "Conduct house-to-house surveys for dengue cases"
            ]
        else:  # Very High Risk
            return base_recommendations + [
                "Establish emergency response teams",
                "Implement mass fogging operations",
                "Set up temporary medical camps",
                "Coordinate with health authorities for outbreak control",
                "Implement travel restrictions if necessary"
            ]

    def predict_comprehensive(self, garbage_count: float, waterlogging_count: float,
                            garbage_text: str = "", waterlogging_text: str = "",
                            ward_id: int = None) -> Dict:
        """
        Comprehensive prediction including risk assessment and recommendations

        Args:
            garbage_count: Number of garbage complaints
            waterlogging_count: Number of waterlogging complaints
            garbage_text: Description of garbage issues
            waterlogging_text: Description of waterlogging issues
            ward_id: Ward identifier

        Returns:
            Dictionary with comprehensive prediction results
        """

        # Prepare features
        features_df = self.prepare_prediction_features(
            garbage_count, waterlogging_count, garbage_text, waterlogging_text, ward_id
        )

        # Make prediction
        predicted_cases = self.predict_dengue_cases(features_df)

        # Classify risk
        risk_level, risk_color, risk_description = self.classify_risk_level(predicted_cases)

        # Get recommendations
        recommendations = self.get_prevention_recommendations(risk_level)

        # Calculate confidence based on input data quality
        input_quality = min(1.0, (garbage_count + waterlogging_count + len(garbage_text + waterlogging_text) / 50) / 10)
        confidence = max(0.3, min(0.9, input_quality))

        return {
            'predicted_cases': round(predicted_cases, 1),
            'risk_level': risk_level,
            'risk_color': risk_color,
            'risk_description': risk_description,
            'recommendations': recommendations,
            'confidence': round(confidence * 100, 1),
            'features_used': {
                'garbage_severity': round(features_df['Garbage_Complaints'].iloc[0], 2),
                'waterlogging_severity': round(features_df['Waterlogging_Complaints'].iloc[0], 2),
                'rainfall_factor': round(features_df['Rainfall_mm'].iloc[0], 1),
                'temperature': round(features_df['Avg_Temp_C'].iloc[0], 1),
                'is_monsoon': bool(features_df['Is_Monsoon'].iloc[0])
            }
        }</content>
<parameter name="filePath">/Users/abhiroyal/Downloads/bang-main/improved_predictor.py