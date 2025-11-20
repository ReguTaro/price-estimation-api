#!/usr/bin/env python3
"""
Flask API for Real Estate Price Estimation using Linear Regression
This API wraps the linear regression functionality for web access
Enhanced with Pearson R correlation coefficient for validation
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Configure CORS to allow requests from InfinityFree and other sources
CORS(app, resources={
    r"/api/*": {
        "origins": "*",  # Allow all origins (InfinityFree has dynamic IPs)
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Accept", "User-Agent"],
        "expose_headers": ["Content-Type"],
        "max_age": 3600
    }
})

# Request logging for debugging
@app.before_request
def log_request_info():
    """Log incoming requests for debugging"""
    app.logger.info('Headers: %s', request.headers)
    app.logger.info('Body: %s', request.get_data())

@app.after_request
def after_request(response):
    """Add security headers and ensure CORS"""
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
    return response

def interpret_pearson_r(r_value):
    """
    Interpret Pearson R correlation coefficient
    
    Returns interpretation string based on correlation strength
    """
    abs_r = abs(r_value)
    
    if abs_r >= 0.90:
        strength = "Very Strong"
    elif abs_r >= 0.70:
        strength = "Strong"
    elif abs_r >= 0.50:
        strength = "Moderate"
    elif abs_r >= 0.30:
        strength = "Weak"
    else:
        strength = "Very Weak"
    
    direction = "Positive" if r_value > 0 else "Negative"
    
    return f"{strength} {direction} Correlation"

def calculate_price_estimation(current_price, property_size, historical_data=None):
    """
    Calculate price estimation using linear regression based on historical data
    
    Args:
        current_price: Current total property price (2025)
        property_size: Property size in square meters
        historical_data: Dictionary of historical TOTAL prices from database (2020-2024)
                        Each value should be: price_per_sqm × property_size
    
    Returns:
        Dictionary containing estimation results
    """
    
    # Use provided historical data or fall back to default Tarlac BIR rates
    if historical_data and len(historical_data) > 0:
        # Use database historical TOTAL prices
        # Convert total prices to price per sqm for regression calculations
        price_data = {}
        for year_str, total_price in historical_data.items():
            price_data[year_str] = float(total_price) / property_size
    else:
        # Fallback to default Tarlac City BIR rates per sqm
        price_data = {
            '2020': 45,
            '2021': 52,
            '2022': 61,
            '2023': 73,
            '2024': 85
        }
    
    # Add current year data (2025) - convert to price per sqm
    price_data['2025'] = current_price / property_size
    
    # Prepare data for linear regression
    years = []
    prices_per_sqm = []
    
    for year_str, price in price_data.items():
        years.append(int(year_str))
        prices_per_sqm.append(float(price))
    
    years = np.array(years).reshape(-1, 1)
    prices_per_sqm = np.array(prices_per_sqm)
    
    # Calculate Pearson correlation coefficient
    # This measures the linear relationship between years and prices
    years_flat = years.flatten()
    pearson_r, p_value = pearsonr(years_flat, prices_per_sqm)
    
    # Normalize years for better regression (start from 0)
    years_normalized = years - years.min()
    
    # Create and train the linear regression model
    model = LinearRegression()
    model.fit(years_normalized, prices_per_sqm)
    
    # Calculate R-squared score for model accuracy
    r_squared = model.score(years_normalized, prices_per_sqm)
    
    # Try polynomial regression if linear fit is poor
    best_model = model
    best_r_squared = r_squared
    model_type = "linear"
    
    if r_squared < 0.85:  # If linear model doesn't fit well
        # Try polynomial regression (degree 2)
        poly = PolynomialFeatures(degree=2)
        years_poly = poly.fit_transform(years_normalized)
        
        poly_model = LinearRegression()
        poly_model.fit(years_poly, prices_per_sqm)
        poly_r_squared = poly_model.score(years_poly, prices_per_sqm)
        
        if poly_r_squared > r_squared + 0.1:  # If polynomial is significantly better
            best_model = poly_model
            best_r_squared = poly_r_squared
            model_type = "polynomial"
    
    # Predict 2026 price per sqm
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
    
    # Calculate historical growth rates to apply momentum
    growth_rate = calculate_growth_rate(prices_per_sqm)
    
    # Calculate recent growth trend (last 2 years weighted more heavily)
    recent_growth = calculate_recent_trend(prices_per_sqm)
    
    # Apply smart market adjustments based on growth momentum
    # Use weighted average of historical growth and linear prediction
    if growth_rate > 15:  # High growth area - maintain momentum
        # Apply 70% of recent growth rate
        momentum_factor = 1 + (recent_growth / 100 * 0.7)
        predicted_2026_price_per_sqm = prices_per_sqm[-1] * momentum_factor
    elif growth_rate > 10:  # Moderate-high growth
        # Apply 60% of recent growth rate
        momentum_factor = 1 + (recent_growth / 100 * 0.6)
        predicted_2026_price_per_sqm = prices_per_sqm[-1] * momentum_factor
    elif growth_rate > 5:  # Moderate growth
        # Apply 50% of recent growth rate
        momentum_factor = 1 + (recent_growth / 100 * 0.5)
        predicted_2026_price_per_sqm = prices_per_sqm[-1] * momentum_factor
    else:  # Low growth - conservative prediction
        # Use 40% of recent growth or minimum 2%
        adjusted_growth = max(recent_growth * 0.4, 2.0)
        momentum_factor = 1 + (adjusted_growth / 100)
        predicted_2026_price_per_sqm = prices_per_sqm[-1] * momentum_factor
    
    # Calculate metrics
    current_price_per_sqm = current_price / property_size
    estimated_total_price = predicted_2026_price_per_sqm * property_size
    profit_projection = estimated_total_price - current_price
    percentage_growth = ((estimated_total_price - current_price) / current_price) * 100
    
    # Get regression coefficients for linear model
    if model_type == "linear":
        slope = model.coef_[0]
        intercept = model.intercept_
    else:
        slope = growth_rate  # Use growth rate for polynomial
        intercept = prices_per_sqm[0]
    
    # Prepare historical data with predictions for chart
    chart_data = []
    for year in range(2020, 2027):
        year_norm = np.array([[year - years.min()]])
        
        if str(year) in price_data and year <= 2025:
            # Use actual historical data
            price = price_data[str(year)]
        else:
            # Use predicted data
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
    
    # Calculate prediction confidence based on data quality and model fit
    # Now includes Pearson R in the confidence calculation
    confidence_score = calculate_confidence(best_r_squared, len(price_data), growth_rate, pearson_r)
    
    # Interpretation of correlation strength
    correlation_interpretation = interpret_pearson_r(pearson_r)
    
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
            'pearson_r': round(pearson_r, 4),
            'p_value': round(p_value, 6),
            'correlation_interpretation': correlation_interpretation,
            'equation': f"y = {round(slope, 2)}x + {round(intercept, 2)}" if model_type == "linear" else "Polynomial (degree 2)"
        },
        'validation': {
            'pearson_correlation': round(pearson_r, 4),
            'correlation_strength': correlation_interpretation,
            'statistical_significance': 'Significant' if p_value < 0.05 else 'Not Significant',
            'p_value': round(p_value, 6),
            'sample_size': len(price_data),
            'note': 'Pearson R measures the strength and direction of linear relationship. Values closer to 1 or -1 indicate stronger correlation.'
        },
        'historical_data': {str(k): v for k, v in price_data.items()},
        'chart_data': chart_data,
        'data_source': 'database' if historical_data else 'default_bir'
    }
    
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

def calculate_recent_trend(prices):
    """
    Calculate recent growth trend with more weight on recent years
    This gives more realistic predictions based on current momentum
    """
    if len(prices) < 3:
        return calculate_growth_rate(prices)
    
    # Calculate growth rates for each year
    growth_rates = []
    for i in range(1, len(prices)):
        if prices[i-1] > 0:
            rate = ((prices[i] - prices[i-1]) / prices[i-1]) * 100
            growth_rates.append(rate)
    
    if not growth_rates:
        return 0
    
    # Weight recent years more heavily
    # Last year: 40%, Second last: 30%, Third last: 20%, Rest: 10%
    if len(growth_rates) >= 3:
        weights = [0.1 / max(1, len(growth_rates) - 3)] * max(0, len(growth_rates) - 3) + [0.2, 0.3, 0.4]
        weights = weights[-len(growth_rates):]  # Trim to actual length
    else:
        # If less than 3 years, distribute evenly but favor recent
        weights = [0.3, 0.7][-len(growth_rates):]
    
    weighted_growth = sum(rate * weight for rate, weight in zip(growth_rates, weights))
    return weighted_growth

def calculate_confidence(r_squared, data_points, growth_rate, pearson_r):
    """
    Calculate confidence score based on multiple factors
    Enhanced with Pearson R correlation coefficient
    """
    # Base confidence from R-squared (max 40 points)
    r2_confidence = r_squared * 40
    
    # Pearson R contribution (max 30 points)
    # Higher absolute value means stronger linear relationship
    pearson_confidence = (abs(pearson_r) ** 2) * 30
    
    # Data points factor (max 15 points)
    data_confidence = min(data_points / 6 * 15, 15)
    
    # Growth rate stability (max 15 points)
    if 5 <= growth_rate <= 15:  # Stable growth
        growth_confidence = 15
    elif growth_rate < 0 or growth_rate > 25:  # Unstable
        growth_confidence = 3
    else:
        growth_confidence = 8
    
    total_confidence = r2_confidence + pearson_confidence + data_confidence + growth_confidence
    
    return round(min(total_confidence, 100))

# API Routes

@app.route('/', methods=['GET'])
def home():
    """Health check endpoint"""
    return jsonify({
        'status': 'online',
        'message': 'Real Estate Price Estimation API with Pearson R Validation',
        'version': '1.1.0',
        'endpoints': {
            '/': 'Health check',
            '/api/estimate': 'POST - Calculate price estimation with Pearson R',
            '/health': 'GET - Health check'
        },
        'new_features': {
            'pearson_r': 'Pearson correlation coefficient for validation',
            'validation': 'Enhanced statistical validation metrics',
            'confidence': 'Improved confidence score using Pearson R'
        }
    })

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'price-estimation-api',
        'version': '1.1.0'
    })

@app.route('/api/estimate', methods=['POST', 'OPTIONS'])
def estimate():
    """
    Calculate price estimation with Pearson R validation
    
    Expected JSON body:
    {
        "current_price": 9500,
        "property_size": 100,
        "historical_data": {
            "2020": 4500,
            "2021": 5200,
            "2022": 6100,
            "2023": 7300,
            "2024": 8500
        }
    }
    
    Note: historical_data should contain TOTAL property prices (price_per_sqm × property_size)
    Example: If property is 100 sqm and 2020 price was ₱45/sqm, send 4500 (45 × 100)
    
    Returns:
    - Standard estimation results
    - Pearson R correlation coefficient
    - P-value for statistical significance
    - Correlation interpretation
    """
    # Handle preflight OPTIONS request
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200
    
    try:
        data = request.get_json()
        
        if not data:
            app.logger.error('No JSON data provided in request')
            return jsonify({
                'success': False,
                'error': 'No JSON data provided',
                'help': 'Send POST request with JSON body containing current_price, property_size, and historical_data'
            }), 400
        
        # Validate required fields
        if 'current_price' not in data or 'property_size' not in data:
            app.logger.error('Missing required fields: %s', data.keys())
            return jsonify({
                'success': False,
                'error': 'Missing required fields: current_price and property_size',
                'received_fields': list(data.keys())
            }), 400
        
        current_price = float(data['current_price'])
        property_size = float(data['property_size'])
        historical_data = data.get('historical_data', None)
        
        # Validate values
        if current_price <= 0 or property_size <= 0:
            return jsonify({
                'success': False,
                'error': 'Price and size must be positive numbers',
                'current_price': current_price,
                'property_size': property_size
            }), 400
        
        # Log successful validation
        app.logger.info('Processing estimation: price=%s, size=%s, historical_years=%s',
                       current_price, property_size, 
                       len(historical_data) if historical_data else 0)
        
        # Calculate estimation
        result = calculate_price_estimation(current_price, property_size, historical_data)
        
        app.logger.info('Estimation successful: estimated_price=%s, pearson_r=%s', 
                       result['estimation']['estimated_price'],
                       result['regression']['pearson_r'])
        
        return jsonify(result)
        
    except ValueError as e:
        app.logger.error('ValueError: %s', str(e))
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Invalid input values provided',
            'error_type': 'ValueError'
        }), 400
        
    except Exception as e:
        app.logger.error('Unexpected error: %s', str(e), exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'An error occurred during calculation',
            'error_type': type(e).__name__
        }), 500

if __name__ == '__main__':
    # Configure logging
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Use PORT environment variable for Railway
    import os
    port = int(os.environ.get('PORT', 5000))
    
    app.logger.info('Starting Flask API on port %s', port)
    app.logger.info('API URL: https://web-production-7f611.up.railway.app')
    app.logger.info('Enhanced with Pearson R correlation validation')
    
    app.run(host='0.0.0.0', port=port, debug=False)
