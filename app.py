#!/usr/bin/env python3
"""
Flask API for Real Estate Price Estimation with Validation
Includes Pearson R validation and backtesting functionality
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from validation import EstimationValidator
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

def calculate_price_estimation(current_price, property_size, historical_data=None, include_validation=True):
    """
    Calculate price estimation with optional validation
    
    Args:
        current_price: Current total property price (2025)
        property_size: Property size in square meters
        historical_data: Dictionary of historical prices from database (2020-2024)
        include_validation: Whether to include validation metrics
    
    Returns:
        Dictionary containing estimation results and validation
    """
    
    # Use provided historical data or fall back to default
    if historical_data and len(historical_data) > 0:
        price_data = historical_data
    else:
        price_data = {
            '2020': 45,
            '2021': 52,
            '2022': 61,
            '2023': 73,
            '2024': 85
        }
    
    # Store original historical data for validation (without 2025)
    original_historical = price_data.copy()
    
    # Add current year data (2025)
    price_data['2025'] = current_price / property_size
    
    # Prepare data for linear regression
    years = []
    prices_per_sqm = []
    
    for year_str, price in price_data.items():
        years.append(int(year_str))
        prices_per_sqm.append(float(price))
    
    years = np.array(years).reshape(-1, 1)
    prices_per_sqm = np.array(prices_per_sqm)
    
    # Normalize years
    years_normalized = years - years.min()
    
    # Create and train model
    model = LinearRegression()
    model.fit(years_normalized, prices_per_sqm)
    
    # Calculate R-squared
    r_squared = model.score(years_normalized, prices_per_sqm)
    
    # Try polynomial if needed
    best_model = model
    best_r_squared = r_squared
    model_type = "linear"
    
    if r_squared < 0.85:
        poly = PolynomialFeatures(degree=2)
        years_poly = poly.fit_transform(years_normalized)
        
        poly_model = LinearRegression()
        poly_model.fit(years_poly, prices_per_sqm)
        poly_r_squared = poly_model.score(years_poly, prices_per_sqm)
        
        if poly_r_squared > r_squared + 0.1:
            best_model = poly_model
            best_r_squared = poly_r_squared
            model_type = "polynomial"
    
    # Predict 2026
    year_2026_normalized = np.array([[2026 - years.min()]])
    
    if model_type == "polynomial":
        poly = PolynomialFeatures(degree=2)
        years_poly = poly.fit_transform(years_normalized)
        poly_model = LinearRegression()
        poly_model.fit(years_poly, prices_per_sqm)
        year_2026_poly = poly.transform(year_2026_normalized)
        predicted_2026_price_per_sqm = poly_model.predict(year_2026_poly)[0]
    else:
        predicted_2026_price_per_sqm = best_model.predict(year_2026_normalized)[0]
    
    # Apply market adjustments
    growth_rate = calculate_growth_rate(prices_per_sqm)
    
    if growth_rate > 15:
        predicted_2026_price_per_sqm *= 1.05
    elif growth_rate < 5:
        predicted_2026_price_per_sqm *= 0.98
    
    # Calculate metrics
    current_price_per_sqm = current_price / property_size
    estimated_total_price = predicted_2026_price_per_sqm * property_size
    profit_projection = estimated_total_price - current_price
    percentage_growth = ((estimated_total_price - current_price) / current_price) * 100
    
    # Get coefficients
    if model_type == "linear":
        slope = model.coef_[0]
        intercept = model.intercept_
    else:
        slope = growth_rate
        intercept = prices_per_sqm[0]
    
    # Prepare chart data
    chart_data = []
    for year in range(2020, 2027):
        year_norm = np.array([[year - years.min()]])
        
        if str(year) in price_data and year <= 2025:
            price = price_data[str(year)]
        else:
            if model_type == "polynomial":
                year_poly = poly.transform(year_norm)
                price = poly_model.predict(year_poly)[0]
            else:
                price = model.predict(year_norm)[0]
        
        chart_data.append({
            'year': year,
            'price_per_sqm': round(float(price), 2),
            'is_historical': year <= 2025
        })
    
    # Calculate confidence
    confidence_score = calculate_confidence(best_r_squared, len(price_data), growth_rate)
    
    # VALIDATION - Use Pearson R to validate model accuracy
    validation_results = None
    if include_validation and len(original_historical) >= 3:
        try:
            validator = EstimationValidator()
            
            # Get comprehensive validation
            validation_results = validator.validate_model_quality(original_historical)
            
            # Get prediction reliability for 2026
            reliability = validator.validate_future_prediction_reliability(
                original_historical, 
                predicted_2026_price_per_sqm
            )
            
            if reliability['success']:
                validation_results['prediction_reliability'] = reliability
                
        except Exception as e:
            print(f"Validation error: {str(e)}")
            validation_results = {
                'success': False,
                'message': f'Validation failed: {str(e)}'
            }
    
    # Prepare response
    result = {
        'success': True,
        'estimation': {
            'current_price': round(current_price, 2),
            'property_size': property_size,
            'current_price_per_sqm': round(current_price_per_sqm, 2),
            'estimated_price': round(estimated_total_price, 2),
            'profit_projection': round(profit_projection, 2),
            'predicted_2026_price_per_sqm': round(predicted_2026_price_per_sqm, 2),
            'percentage_growth': round(percentage_growth, 2),
            'confidence_score': confidence_score,
            'annual_growth_rate': round(growth_rate, 2)
        },
        'regression': {
            'model_type': model_type,
            'slope': round(slope, 4),
            'intercept': round(intercept, 4),
            'r_squared': round(best_r_squared, 4),
            'equation': f"y = {round(slope, 2)}x + {round(intercept, 2)}" if model_type == "linear" else "Polynomial (degree 2)"
        },
        'historical_data': {str(k): v for k, v in price_data.items()},
        'chart_data': chart_data,
        'data_source': 'database' if historical_data else 'default_bir'
    }
    
    # Add validation results if available
    if validation_results:
        result['validation'] = validation_results
    
    return result

def calculate_growth_rate(prices):
    """Calculate average annual growth rate"""
    if len(prices) < 2:
        return 0
    
    growth_rates = []
    for i in range(1, len(prices)):
        if prices[i-1] > 0:
            rate = ((prices[i] - prices[i-1]) / prices[i-1]) * 100
            growth_rates.append(rate)
    
    return np.mean(growth_rates) if growth_rates else 0

def calculate_confidence(r_squared, data_points, growth_rate):
    """Calculate confidence score based on multiple factors"""
    r2_confidence = r_squared * 60
    data_confidence = min(data_points / 6 * 20, 20)
    
    if 5 <= growth_rate <= 15:
        growth_confidence = 20
    elif growth_rate < 0 or growth_rate > 25:
        growth_confidence = 5
    else:
        growth_confidence = 10
    
    total_confidence = r2_confidence + data_confidence + growth_confidence
    
    return round(min(total_confidence, 100))

# API Routes

@app.route('/', methods=['GET'])
def home():
    """Health check endpoint"""
    return jsonify({
        'status': 'online',
        'message': 'Real Estate Price Estimation API with Validation',
        'version': '2.0.0',
        'endpoints': {
            '/': 'Health check',
            '/api/estimate': 'POST - Calculate price estimation (with validation)',
            '/api/validate': 'POST - Validate historical data using Pearson R',
            '/health': 'GET - Health check'
        },
        'features': [
            'Linear & Polynomial Regression',
            'Pearson R Validation',
            'Backtesting',
            'Cross-Validation',
            'Confidence Intervals'
        ]
    })

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'price-estimation-api',
        'version': '2.0.0'
    })

@app.route('/api/estimate', methods=['POST'])
def estimate():
    """
    Calculate price estimation with validation
    
    Expected JSON body:
    {
        "current_price": 1000000,
        "property_size": 100,
        "historical_data": {
            "2020": 45,
            "2021": 52,
            "2022": 61,
            "2023": 73,
            "2024": 85
        },
        "include_validation": true  // Optional, default true
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'No JSON data provided'
            }), 400
        
        # Validate required fields
        if 'current_price' not in data or 'property_size' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing required fields: current_price and property_size'
            }), 400
        
        current_price = float(data['current_price'])
        property_size = float(data['property_size'])
        historical_data = data.get('historical_data', None)
        include_validation = data.get('include_validation', True)
        
        # Validate values
        if current_price <= 0 or property_size <= 0:
            return jsonify({
                'success': False,
                'error': 'Price and size must be positive numbers'
            }), 400
        
        # Calculate estimation with validation
        result = calculate_price_estimation(
            current_price, 
            property_size, 
            historical_data,
            include_validation
        )
        
        return jsonify(result)
        
    except ValueError as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Invalid input values provided'
        }), 400
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'An error occurred during calculation'
        }), 500

@app.route('/api/validate', methods=['POST'])
def validate_only():
    """
    Validate historical data using Pearson R
    
    Expected JSON body:
    {
        "historical_data": {
            "2020": 45,
            "2021": 52,
            "2022": 61,
            "2023": 73,
            "2024": 85
        }
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'historical_data' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing historical_data field'
            }), 400
        
        historical_data = data['historical_data']
        
        if len(historical_data) < 3:
            return jsonify({
                'success': False,
                'error': 'Need at least 3 years of historical data for validation'
            }), 400
        
        # Run validation
        validator = EstimationValidator()
        validation_result = validator.validate_model_quality(historical_data)
        
        return jsonify(validation_result)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Validation failed'
        }), 500

@app.route('/api/backtest', methods=['POST'])
def backtest():
    """
    Backtest model accuracy by holding out 2024
    
    Expected JSON body:
    {
        "historical_data": {
            "2020": 45,
            "2021": 52,
            "2022": 61,
            "2023": 73,
            "2024": 85
        },
        "property_size": 100
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'historical_data' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing historical_data field'
            }), 400
        
        historical_data = data['historical_data']
        property_size = data.get('property_size', 1)  # Default to 1 for per sqm
        
        # Run backtest
        validator = EstimationValidator()
        backtest_result = validator.backtest_single_property(historical_data, property_size)
        
        return jsonify(backtest_result)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Backtest failed'
        }), 500

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
