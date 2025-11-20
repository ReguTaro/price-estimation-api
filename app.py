#!/usr/bin/env python3
"""
Flask API for Real Estate Price Estimation using Linear Regression
This API wraps the linear regression functionality for web access

UPDATED: Variable naming now consistently uses 'historical' prefix
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import warnings
import logging
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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
    logger.info('Headers: %s', request.headers)
    if request.method == 'POST':
        logger.info(f'Body: {request.get_data(as_text=True)}')

@app.after_request
def after_request(response):
    """Add security headers and ensure CORS"""
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
    return response


def calculate_price_estimation(current_price, property_size, historical_data=None):
    """
    Calculate price estimation using linear regression based on historical data
    
    Args:
        current_price (float): Current total property price (2025)
        property_size (float): Property size in square meters
        historical_data (dict): Dictionary of TOTAL prices for years 2020-2024
                               Format: {'2020': 8100, '2021': 9360, ...}
                               Values are TOTAL property prices (not per sqm)
    
    Returns:
        dict: Dictionary containing estimation results with all metrics
    """
    
    try:
        # ================================================================
        # STEP 1: PROCESS historical_data
        # Convert TOTAL prices to PRICE PER SQM
        # ================================================================
        logger.info('=' * 60)
        logger.info('STARTING PRICE ESTIMATION')
        logger.info('=' * 60)
        logger.info(f'Current Price (2025): ₱{current_price:,.2f}')
        logger.info(f'Property Size: {property_size} sqm')
        
        # Process historical_data received from PHP
        if historical_data and len(historical_data) > 0:
            logger.info(f'✓ historical_data received with {len(historical_data)} years')
            logger.info('Converting TOTAL prices to PRICE PER SQM...')
            
            # Convert each total price in historical_data to price per sqm
            historical_prices_per_sqm = {}
            for year_str, historical_total_price in historical_data.items():
                price_per_sqm = float(historical_total_price) / property_size
                historical_prices_per_sqm[year_str] = price_per_sqm
                logger.info(f'  {year_str}: ₱{historical_total_price:,.2f} ÷ {property_size} = ₱{price_per_sqm:.2f}/sqm')
                
        else:
            # No historical_data provided - use default BIR rates
            logger.warning('⚠ No historical_data provided, using default BIR rates')
            historical_prices_per_sqm = {
                '2020': 45,
                '2021': 52,
                '2022': 61,
                '2023': 73,
                '2024': 85
            }
        
        # ================================================================
        # STEP 2: ADD CURRENT YEAR TO historical_prices_per_sqm
        # ================================================================
        current_price_per_sqm = current_price / property_size
        historical_prices_per_sqm['2025'] = current_price_per_sqm
        logger.info(f'Adding 2025: ₱{current_price:,.2f} ÷ {property_size} = ₱{current_price_per_sqm:.2f}/sqm')
        
        # ================================================================
        # STEP 3: PREPARE ARRAYS FROM historical_prices_per_sqm
        # ================================================================
        historical_years = []
        historical_prices = []
        
        for year_str, price_per_sqm in sorted(historical_prices_per_sqm.items()):
            historical_years.append(int(year_str))
            historical_prices.append(float(price_per_sqm))
        
        # Convert to numpy arrays
        historical_years = np.array(historical_years).reshape(-1, 1)
        historical_prices = np.array(historical_prices)
        
        logger.info(f'Historical Years: {historical_years.flatten().tolist()}')
        logger.info(f'Historical Prices (per sqm): {historical_prices.tolist()}')
        
        # ================================================================
        # STEP 4: NORMALIZE historical_years (start from 0)
        # ================================================================
        historical_years_normalized = historical_years - historical_years.min()
        logger.info(f'Normalized Years: {historical_years_normalized.flatten().tolist()}')
        
        # ================================================================
        # STEP 5: LINEAR REGRESSION on historical_data
        # ================================================================
        logger.info('Training Linear Regression model...')
        model = LinearRegression()
        model.fit(historical_years_normalized, historical_prices)
        
        # Calculate R-squared score for model accuracy
        r_squared = model.score(historical_years_normalized, historical_prices)
        slope = model.coef_[0]
        intercept = model.intercept_
        
        logger.info(f'✓ Model trained: y = {slope:.2f}x + {intercept:.2f}')
        logger.info(f'✓ R² Score: {r_squared:.4f} ({r_squared*100:.1f}% variance explained)')
        
        # ================================================================
        # STEP 6: TRY POLYNOMIAL REGRESSION if needed
        # ================================================================
        best_model = model
        best_r_squared = r_squared
        model_type = "linear"
        
        if r_squared < 0.85:  # If linear model doesn't fit well
            logger.info('⚠ R² < 0.85, trying polynomial regression...')
            poly = PolynomialFeatures(degree=2)
            historical_years_poly = poly.fit_transform(historical_years_normalized)
            
            poly_model = LinearRegression()
            poly_model.fit(historical_years_poly, historical_prices)
            poly_r_squared = poly_model.score(historical_years_poly, historical_prices)
            
            logger.info(f'Polynomial R²: {poly_r_squared:.4f}')
            
            if poly_r_squared > r_squared + 0.1:  # If significantly better
                best_model = poly_model
                best_r_squared = poly_r_squared
                model_type = "polynomial"
                logger.info('✓ Using polynomial model (better fit)')
        
        # ================================================================
        # STEP 7: CALCULATE GROWTH RATES from historical_prices
        # ================================================================
        logger.info('Calculating growth rates from historical data...')
        historical_growth_rate = calculate_growth_rate(historical_prices)
        historical_recent_trend = calculate_recent_trend(historical_prices)
        
        logger.info(f'Average Historical Growth Rate: {historical_growth_rate:.2f}%')
        logger.info(f'Recent Weighted Trend: {historical_recent_trend:.2f}%')
        
        # ================================================================
        # STEP 8: PREDICT 2026 PRICE using historical_data patterns
        # ================================================================
        logger.info('Predicting 2026 price...')
        
        # Predict base price using regression
        year_2026_normalized = np.array([[2026 - historical_years.min()[0]]])
        
        if model_type == "polynomial":
            poly = PolynomialFeatures(degree=2)
            poly.fit(historical_years_normalized)
            year_2026_poly = poly.transform(year_2026_normalized)
            predicted_2026_price_per_sqm = best_model.predict(year_2026_poly)[0]
        else:
            predicted_2026_price_per_sqm = best_model.predict(year_2026_normalized)[0]
        
        logger.info(f'Base prediction (regression): ₱{predicted_2026_price_per_sqm:.2f}/sqm')
        
        # ================================================================
        # STEP 9: APPLY MOMENTUM ADJUSTMENT based on historical_growth_rate
        # ================================================================
        logger.info('Applying momentum adjustment...')
        
        if historical_growth_rate > 15:  # High growth area
            momentum_factor = 1 + (historical_recent_trend / 100 * 0.7)
            logger.info(f'High growth detected (>15%), applying 70% momentum')
        elif historical_growth_rate > 10:  # Moderate-high growth
            momentum_factor = 1 + (historical_recent_trend / 100 * 0.6)
            logger.info(f'Moderate-high growth (10-15%), applying 60% momentum')
        elif historical_growth_rate > 5:  # Moderate growth
            momentum_factor = 1 + (historical_recent_trend / 100 * 0.5)
            logger.info(f'Moderate growth (5-10%), applying 50% momentum')
        else:  # Low growth
            adjusted_growth = max(historical_recent_trend * 0.4, 2.0)
            momentum_factor = 1 + (adjusted_growth / 100)
            logger.info(f'Low growth (<5%), using conservative 40% or min 2%')
        
        # Apply momentum to last known price
        predicted_2026_price_per_sqm = historical_prices[-1] * momentum_factor
        
        logger.info(f'Momentum factor: {momentum_factor:.4f}')
        logger.info(f'Final 2026 prediction: ₱{predicted_2026_price_per_sqm:.2f}/sqm')
        
        # ================================================================
        # STEP 10: CALCULATE FINAL METRICS
        # ================================================================
        estimated_total_price = predicted_2026_price_per_sqm * property_size
        profit_projection = estimated_total_price - current_price
        percentage_growth = ((estimated_total_price - current_price) / current_price) * 100
        
        logger.info(f'Estimated 2026 Total Price: ₱{estimated_total_price:,.2f}')
        logger.info(f'Profit Projection: ₱{profit_projection:,.2f}')
        logger.info(f'Percentage Growth: {percentage_growth:.2f}%')
        
        # ================================================================
        # STEP 11: CALCULATE CONFIDENCE SCORE
        # ================================================================
        confidence_score = calculate_confidence(
            best_r_squared, 
            len(historical_prices_per_sqm), 
            historical_growth_rate
        )
        logger.info(f'Confidence Score: {confidence_score}%')
        
        # ================================================================
        # STEP 12: PREPARE CHART DATA from historical_prices_per_sqm
        # ================================================================
        chart_data = []
        for year in range(2020, 2027):
            year_norm = np.array([[year - historical_years.min()[0]]])
            
            if str(year) in historical_prices_per_sqm and year <= 2025:
                # Use actual historical data
                price = historical_prices_per_sqm[str(year)]
                is_historical = True
            else:
                # Use predicted data for 2026
                if model_type == "polynomial":
                    year_poly = poly.transform(year_norm)
                    price = best_model.predict(year_poly)[0]
                else:
                    price = best_model.predict(year_norm)[0]
                # Apply momentum for 2026
                if year == 2026:
                    price = predicted_2026_price_per_sqm
                is_historical = False
            
            chart_data.append({
                'year': year,
                'price_per_sqm': round(float(price), 2),
                'is_historical': is_historical
            })
        
        # ================================================================
        # STEP 13: PREPARE RESPONSE
        # ================================================================
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
                'annual_growth_rate': round(historical_growth_rate, 2)
            },
            'regression': {
                'model_type': model_type,
                'slope': round(slope, 4),
                'intercept': round(intercept, 4),
                'r_squared': round(best_r_squared, 4),
                'equation': f"y = {round(slope, 2)}x + {round(intercept, 2)}" if model_type == "linear" else "Polynomial (degree 2)"
            },
            'historical_data': {str(k): v for k, v in historical_prices_per_sqm.items()},
            'chart_data': chart_data,
            'data_source': 'database' if historical_data else 'default_bir'
        }
        
        logger.info('=' * 60)
        logger.info('ESTIMATION COMPLETED SUCCESSFULLY')
        logger.info('=' * 60)
        
        return result
        
    except Exception as e:
        logger.error(f'Error in calculate_price_estimation: {str(e)}', exc_info=True)
        return {
            'success': False,
            'error': str(e),
            'message': 'An error occurred during calculation'
        }


def calculate_growth_rate(historical_prices):
    """
    Calculate average annual growth rate from historical price data
    
    Args:
        historical_prices: Array of historical prices per sqm
    
    Returns:
        float: Average annual growth rate percentage
    """
    if len(historical_prices) < 2:
        return 0
    
    growth_rates = []
    for i in range(1, len(historical_prices)):
        if historical_prices[i-1] > 0:
            rate = ((historical_prices[i] - historical_prices[i-1]) / historical_prices[i-1]) * 100
            growth_rates.append(rate)
    
    return np.mean(growth_rates) if growth_rates else 0


def calculate_recent_trend(historical_prices):
    """
    Calculate recent growth trend with more weight on recent years
    Uses historical price data to determine current momentum
    
    Args:
        historical_prices: Array of historical prices per sqm
    
    Returns:
        float: Weighted recent growth rate percentage
    """
    if len(historical_prices) < 3:
        return calculate_growth_rate(historical_prices)
    
    # Calculate growth rates for each year
    growth_rates = []
    for i in range(1, len(historical_prices)):
        if historical_prices[i-1] > 0:
            rate = ((historical_prices[i] - historical_prices[i-1]) / historical_prices[i-1]) * 100
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


def calculate_confidence(r_squared, data_points, growth_rate):
    """
    Calculate confidence score based on multiple factors
    
    Args:
        r_squared: R² score from regression model
        data_points: Number of historical data points
        growth_rate: Average growth rate percentage
    
    Returns:
        int: Confidence score (0-100)
    """
    # Base confidence from R-squared (max 60 points)
    r2_confidence = r_squared * 60
    
    # Data points factor (max 20 points)
    data_confidence = min(data_points / 6 * 20, 20)
    
    # Growth rate stability (max 20 points)
    if 5 <= growth_rate <= 15:  # Stable growth
        growth_confidence = 20
    elif growth_rate < 0 or growth_rate > 25:  # Unstable
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
        'message': 'Real Estate Price Estimation API',
        'version': '1.0.1',
        'endpoints': {
            '/': 'Health check',
            '/api/estimate': 'POST - Calculate price estimation',
            '/health': 'GET - Health check'
        },
        'note': 'Updated with improved historical_data variable naming'
    })


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'price-estimation-api'
    })


@app.route('/api/estimate', methods=['POST', 'OPTIONS'])
def estimate():
    """
    Calculate price estimation using historical data
    
    Expected JSON body:
    {
        "current_price": 8100,
        "property_size": 90,
        "historical_data": {
            "2020": 8100,
            "2021": 9360,
            "2022": 10980,
            "2023": 13140,
            "2024": 15300
        }
    }
    
    Note: historical_data should contain TOTAL property prices (price_per_sqm × property_size)
    Example: If property is 90 sqm and 2020 price was ₱90/sqm, send 8100 (90 × 90)
    """
    # Handle preflight OPTIONS request
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200
    
    try:
        data = request.get_json()
        
        if not data:
            logger.error('No JSON data provided in request')
            return jsonify({
                'success': False,
                'error': 'No JSON data provided',
                'help': 'Send POST request with JSON body containing current_price, property_size, and historical_data'
            }), 400
        
        # Validate required fields
        if 'current_price' not in data or 'property_size' not in data:
            logger.error('Missing required fields: %s', data.keys())
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
        logger.info('Processing estimation: price=%s, size=%s, historical_years=%s',
                   current_price, property_size, 
                   len(historical_data) if historical_data else 0)
        
        # Calculate estimation using historical_data
        result = calculate_price_estimation(current_price, property_size, historical_data)
        
        if result['success']:
            logger.info('✓ Estimation successful: estimated_price=%s', 
                       result['estimation']['estimated_price'])
        
        return jsonify(result)
        
    except ValueError as e:
        logger.error('ValueError: %s', str(e))
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Invalid input values provided',
            'error_type': 'ValueError'
        }), 400
        
    except Exception as e:
        logger.error('Unexpected error: %s', str(e), exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'An error occurred during calculation',
            'error_type': type(e).__name__
        }), 500


if __name__ == '__main__':
    # Use PORT environment variable for Railway
    import os
    port = int(os.environ.get('PORT', 5000))
    
    logger.info('=' * 60)
    logger.info('STARTING FLASK API')
    logger.info('=' * 60)
    logger.info('Port: %s', port)
    logger.info('API URL: https://web-production-7f611.up.railway.app')
    logger.info('Version: 1.0.1 (Updated variable naming)')
    logger.info('=' * 60)
    
    app.run(host='0.0.0.0', port=port, debug=False)
