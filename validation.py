#!/usr/bin/env python3
"""
Validation Module for Real Estate Price Estimation
Implements Pearson R correlation and backtesting for model accuracy validation
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from scipy.stats import pearsonr
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class EstimationValidator:
    """
    Validates price estimation accuracy using backtesting and Pearson R correlation
    """
    
    def __init__(self):
        self.validation_results = []
    
    def backtest_single_property(self, historical_data: Dict, property_size: float) -> Dict:
        """
        Backtest a single property by holding out the most recent year (2024)
        Train on 2020-2023, predict 2024, compare with actual
        
        Args:
            historical_data: Dictionary with years as keys, price_per_sqm as values
            property_size: Property size in square meters
            
        Returns:
            Dictionary with validation metrics
        """
        # Ensure we have enough data
        if len(historical_data) < 4:
            return {
                'success': False,
                'message': 'Insufficient historical data for backtesting (need at least 4 years)'
            }
        
        # Split data: train on 2020-2023, test on 2024
        train_years = []
        train_prices = []
        
        for year_str, price_per_sqm in historical_data.items():
            year = int(year_str)
            if year < 2024:  # Training data
                train_years.append(year)
                train_prices.append(float(price_per_sqm))
        
        # Get actual 2024 value if it exists
        actual_2024 = historical_data.get('2024', None)
        
        if actual_2024 is None:
            return {
                'success': False,
                'message': 'No 2024 data available for validation'
            }
        
        actual_2024 = float(actual_2024)
        
        # Train model on 2020-2023
        train_years = np.array(train_years).reshape(-1, 1)
        train_prices = np.array(train_prices)
        
        # Normalize years
        years_normalized = train_years - train_years.min()
        
        # Linear regression
        model = LinearRegression()
        model.fit(years_normalized, train_prices)
        
        # Predict 2024
        year_2024_normalized = np.array([[2024 - train_years.min()]])
        predicted_2024 = model.predict(year_2024_normalized)[0]
        
        # Calculate errors
        absolute_error = abs(predicted_2024 - actual_2024)
        percentage_error = (absolute_error / actual_2024) * 100
        
        # Calculate R-squared on training data
        r_squared = model.score(years_normalized, train_prices)
        
        return {
            'success': True,
            'validation_type': 'single_year_holdout',
            'train_years': '2020-2023',
            'test_year': 2024,
            'actual_2024_per_sqm': round(actual_2024, 2),
            'predicted_2024_per_sqm': round(predicted_2024, 2),
            'absolute_error': round(absolute_error, 2),
            'percentage_error': round(percentage_error, 2),
            'r_squared_training': round(r_squared, 4),
            'accuracy': round(100 - percentage_error, 2)
        }
    
    def calculate_pearson_r_validation(self, historical_data: Dict) -> Dict:
        """
        Calculate Pearson R between predicted and actual values
        using leave-one-out cross-validation
        
        Args:
            historical_data: Dictionary with years as keys, price_per_sqm as values
            
        Returns:
            Dictionary with Pearson R and validation metrics
        """
        if len(historical_data) < 3:
            return {
                'success': False,
                'message': 'Insufficient data for Pearson R validation (need at least 3 years)'
            }
        
        years_list = sorted([int(year) for year in historical_data.keys()])
        
        actual_values = []
        predicted_values = []
        
        # Leave-one-out cross-validation
        for holdout_year in years_list:
            # Use all other years for training
            train_years = []
            train_prices = []
            
            for year in years_list:
                if year != holdout_year:
                    year_str = str(year)
                    train_years.append(year)
                    train_prices.append(float(historical_data[year_str]))
            
            if len(train_years) < 2:
                continue
            
            # Train model
            train_years_array = np.array(train_years).reshape(-1, 1)
            train_prices_array = np.array(train_prices)
            
            years_normalized = train_years_array - train_years_array.min()
            
            model = LinearRegression()
            model.fit(years_normalized, train_prices_array)
            
            # Predict holdout year
            holdout_normalized = np.array([[holdout_year - train_years_array.min()]])
            predicted_price = model.predict(holdout_normalized)[0]
            
            # Store actual and predicted
            actual_values.append(float(historical_data[str(holdout_year)]))
            predicted_values.append(predicted_price)
        
        if len(actual_values) < 2:
            return {
                'success': False,
                'message': 'Insufficient predictions for Pearson R calculation'
            }
        
        # Calculate Pearson R
        pearson_r, p_value = pearsonr(actual_values, predicted_values)
        
        # Calculate other metrics
        errors = [abs(pred - actual) for pred, actual in zip(predicted_values, actual_values)]
        mean_absolute_error = np.mean(errors)
        percentage_errors = [(err / actual) * 100 for err, actual in zip(errors, actual_values)]
        mean_percentage_error = np.mean(percentage_errors)
        
        return {
            'success': True,
            'validation_type': 'leave_one_out_cross_validation',
            'pearson_r': round(pearson_r, 4),
            'pearson_r_squared': round(pearson_r ** 2, 4),
            'p_value': round(p_value, 6),
            'significance': 'significant' if p_value < 0.05 else 'not_significant',
            'mean_absolute_error': round(mean_absolute_error, 2),
            'mean_percentage_error': round(mean_percentage_error, 2),
            'prediction_accuracy': round(100 - mean_percentage_error, 2),
            'correlation_strength': self._interpret_pearson_r(pearson_r),
            'number_of_predictions': len(actual_values),
            'predictions': [
                {
                    'actual': round(actual, 2),
                    'predicted': round(pred, 2),
                    'error': round(abs(pred - actual), 2)
                }
                for actual, pred in zip(actual_values, predicted_values)
            ]
        }
    
    def _interpret_pearson_r(self, r: float) -> str:
        """Interpret Pearson R correlation strength"""
        abs_r = abs(r)
        if abs_r >= 0.9:
            return 'Very Strong'
        elif abs_r >= 0.7:
            return 'Strong'
        elif abs_r >= 0.5:
            return 'Moderate'
        elif abs_r >= 0.3:
            return 'Weak'
        else:
            return 'Very Weak'
    
    def validate_model_quality(self, historical_data: Dict) -> Dict:
        """
        Comprehensive validation combining multiple methods
        
        Args:
            historical_data: Dictionary with years as keys, price_per_sqm as values
            
        Returns:
            Complete validation report
        """
        results = {
            'success': True,
            'validation_date': np.datetime64('today').astype(str),
            'data_years': sorted([int(year) for year in historical_data.keys()]),
            'validations': {}
        }
        
        # Method 1: Single year holdout (2024)
        backtest = self.backtest_single_property(historical_data, property_size=1)  # Use 1 for per sqm
        if backtest['success']:
            results['validations']['backtest_2024'] = backtest
        
        # Method 2: Pearson R with cross-validation
        pearson_validation = self.calculate_pearson_r_validation(historical_data)
        if pearson_validation['success']:
            results['validations']['pearson_r_validation'] = pearson_validation
        
        # Overall assessment
        if pearson_validation['success']:
            pearson_r = pearson_validation['pearson_r']
            accuracy = pearson_validation['prediction_accuracy']
            
            if pearson_r >= 0.9 and accuracy >= 90:
                overall = 'Excellent'
            elif pearson_r >= 0.7 and accuracy >= 80:
                overall = 'Good'
            elif pearson_r >= 0.5 and accuracy >= 70:
                overall = 'Fair'
            else:
                overall = 'Poor'
            
            results['overall_model_quality'] = overall
            results['reliability_score'] = round((pearson_r * 50 + accuracy * 0.5), 2)
        
        return results
    
    def validate_future_prediction_reliability(
        self, 
        historical_data: Dict, 
        predicted_2026_per_sqm: float
    ) -> Dict:
        """
        Estimate the reliability of a future prediction (2026)
        based on historical validation performance
        
        Args:
            historical_data: Historical price per sqm data
            predicted_2026_per_sqm: The predicted 2026 price per sqm
            
        Returns:
            Reliability assessment
        """
        # Run validation
        validation = self.calculate_pearson_r_validation(historical_data)
        
        if not validation['success']:
            return {
                'success': False,
                'message': 'Could not validate prediction reliability'
            }
        
        pearson_r = validation['pearson_r']
        mean_error = validation['mean_percentage_error']
        
        # Calculate confidence interval based on historical errors
        error_margin = predicted_2026_per_sqm * (mean_error / 100)
        
        lower_bound = predicted_2026_per_sqm - error_margin
        upper_bound = predicted_2026_per_sqm + error_margin
        
        return {
            'success': True,
            'predicted_2026_per_sqm': round(predicted_2026_per_sqm, 2),
            'prediction_reliability': validation['correlation_strength'],
            'expected_error_percentage': round(mean_error, 2),
            'confidence_interval': {
                'lower': round(lower_bound, 2),
                'upper': round(upper_bound, 2),
                'margin': round(error_margin, 2)
            },
            'pearson_r': round(pearson_r, 4),
            'recommendation': self._get_prediction_recommendation(pearson_r, mean_error)
        }
    
    def _get_prediction_recommendation(self, pearson_r: float, mean_error: float) -> str:
        """Generate recommendation based on validation metrics"""
        if pearson_r >= 0.9 and mean_error < 5:
            return "High confidence - Model shows excellent predictive accuracy"
        elif pearson_r >= 0.7 and mean_error < 10:
            return "Good confidence - Model is reliable for price estimation"
        elif pearson_r >= 0.5 and mean_error < 15:
            return "Moderate confidence - Use estimation with caution"
        else:
            return "Low confidence - Consider additional data or expert consultation"


def validate_estimation_with_pearson_r(historical_data: Dict, property_size: float) -> Dict:
    """
    Convenience function to validate estimation using Pearson R
    
    Args:
        historical_data: Dictionary with years as keys, price_per_sqm as values
        property_size: Property size in sqm
        
    Returns:
        Validation results
    """
    validator = EstimationValidator()
    return validator.validate_model_quality(historical_data)


if __name__ == "__main__":
    # Test with sample data
    sample_data = {
        '2020': 45,
        '2021': 52,
        '2022': 61,
        '2023': 73,
        '2024': 85
    }
    
    validator = EstimationValidator()
    
    print("=== Validation Test ===\n")
    
    # Test backtesting
    backtest = validator.backtest_single_property(sample_data, property_size=100)
    print("Backtest Results:")
    print(f"Predicted 2024: ₱{backtest.get('predicted_2024_per_sqm', 'N/A')}/sqm")
    print(f"Actual 2024: ₱{backtest.get('actual_2024_per_sqm', 'N/A')}/sqm")
    print(f"Accuracy: {backtest.get('accuracy', 'N/A')}%\n")
    
    # Test Pearson R validation
    pearson = validator.calculate_pearson_r_validation(sample_data)
    print("Pearson R Validation:")
    print(f"Pearson R: {pearson.get('pearson_r', 'N/A')}")
    print(f"Correlation: {pearson.get('correlation_strength', 'N/A')}")
    print(f"Prediction Accuracy: {pearson.get('prediction_accuracy', 'N/A')}%\n")
    
    # Full validation
    full_validation = validator.validate_model_quality(sample_data)
    print("Overall Model Quality:", full_validation.get('overall_model_quality', 'N/A'))
    print("Reliability Score:", full_validation.get('reliability_score', 'N/A'))
