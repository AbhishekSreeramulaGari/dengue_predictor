"""
Ward-Level Analysis Module
Provides utilities for analyzing dengue predictions at the ward level
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple


class WardAnalyzer:
    """Analyzes ward-level prediction performance and patterns"""
    
    def __init__(self):
        """Initialize ward analyzer"""
        self.ward_metrics = {}
        self.performance_tiers = {
            'Excellent': (90, 100),
            'Good': (80, 90),
            'Fair': (70, 80),
            'Poor': (0, 70)
        }
    
    def calculate_ward_metrics(self, predictions_df: pd.DataFrame) -> Dict:
        """
        Calculate performance metrics for each ward
        
        Args:
            predictions_df: DataFrame with columns [Ward_ID, Predicted_Cases, Actual_Cases, Model]
            
        Returns:
            Dictionary with ward-level metrics
        """
        metrics = {}
        
        for ward_id in predictions_df['Ward_ID'].unique():
            ward_data = predictions_df[predictions_df['Ward_ID'] == ward_id]
            
            # Calculate metrics
            predicted = ward_data['Predicted_Cases'].values
            actual = ward_data['Actual_Cases'].values
            
            mae = np.mean(np.abs(predicted - actual))
            rmse = np.sqrt(np.mean((predicted - actual) ** 2))
            mape = np.mean(np.abs((actual - predicted) / (actual + 1)) * 100)
            
            # Calculate accuracy (within 10% error considered accurate)
            accuracy = np.mean(np.abs((actual - predicted) / (actual + 1)) <= 0.1) * 100
            
            # Calculate R²
            ss_res = np.sum((actual - predicted) ** 2)
            ss_tot = np.sum((actual - np.mean(actual)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            metrics[ward_id] = {
                'MAE': mae,
                'RMSE': rmse,
                'MAPE': mape,
                'Accuracy': accuracy,
                'R2': r2,
                'Sample_Size': len(ward_data),
                'Avg_Predicted': np.mean(predicted),
                'Avg_Actual': np.mean(actual)
            }
        
        self.ward_metrics = metrics
        return metrics
    
    def classify_ward_performance(self, accuracy: float) -> str:
        """
        Classify ward performance into tiers
        
        Args:
            accuracy: Accuracy percentage (0-100)
            
        Returns:
            Performance tier string
        """
        for tier, (min_acc, max_acc) in self.performance_tiers.items():
            if min_acc <= accuracy < max_acc:
                return tier
        return 'Poor'
    
    def get_top_wards(self, metric: str = 'Accuracy', n: int = 10) -> List[Tuple]:
        """
        Get top performing wards
        
        Args:
            metric: Metric to sort by (Accuracy, MAE, RMSE, R2)
            n: Number of wards to return
            
        Returns:
            List of (ward_id, metric_value) tuples
        """
        if not self.ward_metrics:
            return []
        
        sorted_wards = sorted(
            self.ward_metrics.items(),
            key=lambda x: x[1][metric],
            reverse=(metric in ['Accuracy', 'R2'])
        )
        
        return [(ward, metrics[metric]) for ward, metrics in sorted_wards[:n]]
    
    def get_bottom_wards(self, metric: str = 'Accuracy', n: int = 10) -> List[Tuple]:
        """
        Get bottom performing wards
        
        Args:
            metric: Metric to sort by
            n: Number of wards to return
            
        Returns:
            List of (ward_id, metric_value) tuples
        """
        if not self.ward_metrics:
            return []
        
        sorted_wards = sorted(
            self.ward_metrics.items(),
            key=lambda x: x[1][metric],
            reverse=(metric in ['Accuracy', 'R2'])
        )
        
        return [(ward, metrics[metric]) for ward, metrics in sorted_wards[-n:]]
    
    def get_tier_distribution(self) -> Dict[str, int]:
        """
        Get distribution of wards across performance tiers
        
        Returns:
            Dictionary with tier names and ward counts
        """
        tier_counts = {tier: 0 for tier in self.performance_tiers.keys()}
        
        for ward, metrics in self.ward_metrics.items():
            tier = self.classify_ward_performance(metrics['Accuracy'])
            tier_counts[tier] += 1
        
        return tier_counts
    
    def get_ward_summary(self) -> Dict:
        """
        Get overall summary statistics
        
        Returns:
            Dictionary with summary statistics
        """
        if not self.ward_metrics:
            return {}
        
        accuracies = [m['Accuracy'] for m in self.ward_metrics.values()]
        maes = [m['MAE'] for m in self.ward_metrics.values()]
        rmses = [m['RMSE'] for m in self.ward_metrics.values()]
        r2s = [m['R2'] for m in self.ward_metrics.values()]
        
        return {
            'Total_Wards': len(self.ward_metrics),
            'Mean_Accuracy': float(np.mean(accuracies)),
            'Median_Accuracy': float(np.median(accuracies)),
            'Std_Accuracy': float(np.std(accuracies)),
            'Min_Accuracy': float(np.min(accuracies)),
            'Max_Accuracy': float(np.max(accuracies)),
            'Mean_MAE': float(np.mean(maes)),
            'Mean_RMSE': float(np.mean(rmses)),
            'Mean_R2': float(np.mean(r2s))
        }
    
    def get_model_recommendations(self) -> Dict[str, str]:
        """
        Get model recommendations based on ward performance
        
        Returns:
            Dictionary with recommendations
        """
        summary = self.get_ward_summary()
        tier_dist = self.get_tier_distribution()
        
        avg_accuracy = summary.get('Mean_Accuracy', 0)
        
        recommendations = {
            'Overall': '',
            'Model_Selection': '',
            'Improvement_Priority': ''
        }
        
        if avg_accuracy >= 85:
            recommendations['Overall'] = 'System is production-ready with excellent performance'
            recommendations['Model_Selection'] = 'Continue with current ensemble approach'
        elif avg_accuracy >= 80:
            recommendations['Overall'] = 'System performs well, minor optimizations possible'
            recommendations['Model_Selection'] = 'Consider ensemble methods for improved accuracy'
        else:
            recommendations['Overall'] = 'System needs optimization before production deployment'
            recommendations['Model_Selection'] = 'Implement ensemble methods and feature engineering'
        
        poor_wards = tier_dist.get('Poor', 0)
        if poor_wards > 0:
            recommendations['Improvement_Priority'] = f'Focus on {poor_wards} wards with poor performance'
        else:
            recommendations['Improvement_Priority'] = 'All wards performing adequately'
        
        return recommendations
    
    def get_ward_status(self, ward_id: int) -> Dict:
        """
        Get detailed status for a specific ward
        
        Args:
            ward_id: Ward ID
            
        Returns:
            Dictionary with ward status and metrics
        """
        if ward_id not in self.ward_metrics:
            return {'error': f'Ward {ward_id} not found'}
        
        metrics = self.ward_metrics[ward_id]
        tier = self.classify_ward_performance(metrics['Accuracy'])
        
        tier_colors = {
            'Excellent': 'success',
            'Good': 'info',
            'Fair': 'warning',
            'Poor': 'danger'
        }
        
        return {
            'Ward_ID': ward_id,
            'Performance_Tier': tier,
            'Tier_Color': tier_colors.get(tier, 'secondary'),
            'Accuracy': f"{metrics['Accuracy']:.2f}%",
            'MAE': f"{metrics['MAE']:.4f}",
            'RMSE': f"{metrics['RMSE']:.4f}",
            'R2': f"{metrics['R2']:.4f}",
            'Avg_Predicted_Cases': f"{metrics['Avg_Predicted']:.1f}",
            'Avg_Actual_Cases': f"{metrics['Avg_Actual']:.1f}",
            'Samples': metrics['Sample_Size']
        }


# Create global analyzer instance
analyzer = WardAnalyzer()
