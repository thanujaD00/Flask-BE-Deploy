from flask import Flask, request, jsonify
import pandas as pd
from datetime import datetime
import os
import json
import pickle
import numpy as np
from utils.model_utils import ensemble_predict, load_seasonal_factors
import joblib
from utils.data_utils import analyze_seasonal_patterns, create_seasonality_plot
import time
from functools import lru_cache

# Replace this code at the app initialization
app = Flask(__name__)

with open('models/coconut_price_model.pkl', 'rb') as file:
    model_data = pickle.load(file)
    
# Load all required models directly
try:
    agro_model = joblib.load('models/xgboost_model.pkl')
    ts_model = joblib.load('models/best_time_series_model.pkl')
    ensemble_model = joblib.load('models/weighted_ensemble_model.pkl')  # Linear regression ensemble
    
    print("All models loaded successfully")
except Exception as e:
    print(f"Error loading models: {e}")
    agro_model = None
    ts_model = None  
    ensemble_model = None
    

def predict_coconut_yield(soil_data, weather_data, prediction_date):
    """
    Predict coconut yield using ensemble approach
    """
    if not all([agro_model, ts_model, ensemble_model]):
        raise Exception("Models not loaded properly")
    
    try:
        # Prepare agronomical features
        agro_features = pd.DataFrame({
            'Temperature (°C)': [weather_data['Temperature (°C)']],
            'Humidity (%)': [weather_data['Humidity (%)']],
            'Rainfall (mm)': [weather_data['Rainfall (mm)']],
            'Plant Age (years)': [soil_data['age']],
            'Soil Type': [soil_data['soil_type']]
        })
        
        # Get agronomical prediction
        agro_prediction = agro_model.predict(agro_features)[0]
        
        # For time series prediction, create features based on date
        # This depends on your time series model structure
        ts_features = pd.DataFrame({
            'year': [prediction_date.year],
            'month': [prediction_date.month],
            'day': [prediction_date.day]
        })
        
        # Get time series prediction
        ts_prediction = ts_model.predict(ts_features)[0]
        
        # Combine predictions for ensemble (this uses the linear regression ensemble)
        combined_features = np.array([[agro_prediction, ts_prediction]])
        
        # Get final ensemble prediction
        final_prediction = ensemble_model.predict(combined_features)[0]
        
        return {
            'agronomical_prediction': float(agro_prediction),
            'time_series_prediction': float(ts_prediction),
            'ensemble_prediction': float(final_prediction),
            'prediction_date': prediction_date.strftime('%Y-%m-%d')
        }
        
    except Exception as e:
        raise Exception(f"Prediction error: {str(e)}")
    

# Load model at startup - OUTSIDE of any route handlers
try:
    model_data = joblib.load('models/coconut_price_model.pkl')
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model_data = None

@lru_cache(maxsize=128)
def predict_future_price(yield_nuts, export_volume, domestic_consumption, inflation_rate, 
                         prediction_date, previous_prices=None):
    """
    Predict coconut price for a future date using saved model
    
    Parameters:
    - yield_nuts: Monthly yield in million nuts
    - export_volume: Export volume in million nuts
    - domestic_consumption: Domestic consumption in million nuts
    - inflation_rate: Current inflation rate
    - prediction_date: Date for prediction (string 'YYYY-MM-DD' or datetime)
    - previous_prices: Dictionary with previous 12 months of prices (optional)
    """
    
    # Convert dictionary to tuple for caching if needed
    previous_prices_tuple = tuple(previous_prices.items()) if previous_prices else None
    
    # Convert prediction_date to datetime if string
    if isinstance(prediction_date, str):
        prediction_date = pd.to_datetime(prediction_date)
    
    # Create a single row dataframe with all required features
    input_data = pd.DataFrame({
        'Year': [prediction_date.year],
        'Month': [prediction_date.month],
        'Yield_Million_Nuts': [yield_nuts],
        'Export_Volume_Million_Nuts': [export_volume],
        'Domestic_Consumption_Million_Nuts': [domestic_consumption],
        'Inflation_Rate': [inflation_rate]
    })
    
    # Calculate derived features
    input_data['Year_Sin'] = np.sin(2 * np.pi * input_data['Year']/2025)  # Assuming max year is 2025
    input_data['Month_Sin'] = np.sin(2 * np.pi * input_data['Month']/12)
    input_data['Month_Cos'] = np.cos(2 * np.pi * input_data['Month']/12)
    
    # Supply-Demand Ratio
    input_data['Supply_Demand_Ratio'] = yield_nuts / (export_volume + domestic_consumption)
    
    # If previous prices are provided, calculate lag features
    if previous_prices:
        for lag in [1, 3, 6, 12]:
            input_data[f'Price_Lag_{lag}'] = previous_prices.get(lag, 0)
    else:
        for lag in [1, 3, 6, 12]:
            input_data[f'Price_Lag_{lag}'] = 0
    
    # Add lag features for yield and inflation (using current values as proxies)
    for lag in [1, 3, 6, 12]:
        input_data[f'Yield_Lag_{lag}'] = yield_nuts
        input_data[f'Inflation_Lag_{lag}'] = inflation_rate
    
    # Calculate moving averages (using current values as proxies)
    for window in [3, 6, 12]:
        input_data[f'Yield_MA_{window}'] = yield_nuts
        input_data[f'Export_MA_{window}'] = export_volume
        input_data[f'Domestic_MA_{window}'] = domestic_consumption
        input_data[f'Inflation_MA_{window}'] = inflation_rate
    
    # Ensure all required features are present
    missing_features = set(model_data['features']) - set(input_data.columns)
    for feature in missing_features:
        input_data[feature] = 0
    
    # Reorder columns to match training data
    input_data = input_data[model_data['features']]
    
    # Scale the features
    input_scaled = model_data['scaler'].transform(input_data)
    
        # Make prediction
    prediction = model_data['model'].predict(input_scaled)[0]
    
    return float(prediction)
    


@app.route('/predict', methods=['POST'])
def predict():
    start_time = time.time()
    try:
        data = request.get_json()
        
        # Get prediction year
        prediction_year = data.get('year', datetime.now().year)
        
        # Get monthly data
        monthly_data = data.get('monthly_data', [])
        
        if not monthly_data:
            return jsonify({
                'status': 'error',
                'message': 'No monthly data provided'
            }), 400
        
        # Make predictions for each month
        all_predictions = []
        for month_data in monthly_data:
            try:
                # Extract month
                month = month_data.get('month')
                if not month or not (1 <= month <= 12):
                    return jsonify({
                        'status': 'error',
                        'message': f'Invalid month: {month}'
                    }), 400
                
                # Extract required data based on your original features
                soil_data = {
                    'age': month_data.get('age'),
                    'soil_type': month_data.get('soil_type')
                }
                
                weather_data = {
                    'Temperature (°C)': month_data.get('Temperature (°C)'),
                    'Humidity (%)': month_data.get('Humidity (%)'),
                    'Rainfall (mm)': month_data.get('Rainfall (mm)')
                }

                # Validate required inputs
                required_fields = {
                    'age': soil_data['age'],
                    'soil_type': soil_data['soil_type'],
                    'Temperature (°C)': weather_data['Temperature (°C)'],
                    'Humidity (%)': weather_data['Humidity (%)'],
                    'Rainfall (mm)': weather_data['Rainfall (mm)']
                }
                
                for field_name, field_value in required_fields.items():
                    if field_value is None:
                        return jsonify({
                            'status': 'error',
                            'message': f'Missing parameter for month {month}: {field_name}'
                        }), 400
                
                # Create prediction date
                prediction_date = pd.Timestamp(year=prediction_year, month=month, day=15)
                
                # Make prediction using ensemble approach
                prediction = predict_coconut_yield(soil_data, weather_data, prediction_date)
                prediction['month'] = month
                all_predictions.append(prediction)
                
            except Exception as e:
                return jsonify({
                    'status': 'error',
                    'message': f'Error processing month {month}: {str(e)}'
                }), 400

        if all_predictions:
            print(f"Prediction completed in {time.time() - start_time:.2f} seconds")
            return jsonify({
                'status': 'success',
                'year': prediction_year,
                'monthly_predictions': all_predictions,
                'average_prediction': round(
                    sum(p['ensemble_prediction'] for p in all_predictions) / len(all_predictions), 
                    2
                )
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'Failed to generate predictions'
            }), 500
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/predict_price', methods=['POST'])
def predictPrice():
    data = request.json
    yield_nuts = data.get('yield_nuts')
    export_volume = data.get('export_volume')
    domestic_consumption = data.get('domestic_consumption')
    inflation_rate = data.get('inflation_rate')
    prediction_date = data.get('prediction_date')
    previous_prices = data.get('previous_prices', None)
    
    if not all([yield_nuts, export_volume, domestic_consumption, inflation_rate, prediction_date]):
        return jsonify({'error': 'Missing required parameters'}), 400
    
    predicted_price = predict_future_price(
        yield_nuts=yield_nuts,
        export_volume=export_volume,
        domestic_consumption=domestic_consumption,
        inflation_rate=inflation_rate,
        prediction_date=prediction_date,
        previous_prices=previous_prices
    )
    
    return jsonify({'predicted_price': predicted_price,
                    'date': prediction_date})

@app.route('/analyze_seasonality', methods=['GET'])
def analyze_seasonality():
    try:
        # Path to your historical data
        historical_data_path = 'data/processed_coconut_data.csv'
        
        # Check if the file exists
        if not os.path.exists(historical_data_path):
            return jsonify({
                'status': 'error',
                'message': 'Historical data file not found'
            }), 404
            
        # Perform the analysis
        analysis = analyze_seasonal_patterns(historical_data_path)
        
        if analysis:
            return jsonify({
                'status': 'success',
                'analysis': analysis
            })
            
        return jsonify({
            'status': 'error',
            'message': 'Failed to analyze seasonal patterns'
        }), 500
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/visualize_seasonality', methods=['GET'])
def visualize_seasonality():
    try:
        # Path to your historical data
        historical_data_path = 'data/processed_coconut_data.csv'
        
        # Check if the file exists
        if not os.path.exists(historical_data_path):
            return jsonify({
                'status': 'error',
                'message': 'Historical data file not found'
            }), 404
            
        # Create the visualization
        image = create_seasonality_plot(historical_data_path)
        
        if image:
            return jsonify({
                'status': 'success',
                'image': image
            })
            
        return jsonify({
            'status': 'error',
            'message': 'Failed to create visualization'
        }), 500
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/seasonal_factors', methods=['GET'])
def get_seasonal_factors():
    try:
        seasonal_factors = load_seasonal_factors()
        return jsonify({
            'status': 'success',
            'seasonal_factors': seasonal_factors
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    # Ensure required directories exist
    os.makedirs('models', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    # Set up proper logging
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Pre-load and warm up models
    print("Initializing models...")
    
    # Create example data file if it doesn't exist
    historical_data_path = 'data/processed_coconut_data.csv'
    if not os.path.exists(historical_data_path):
        # Create a sample file based on your provided data
        sample_data = """Date,Soil Moisture (10 cm) (%),Soil Moisture (20 cm) (%),Soil Moisture (30 cm) (%),Plant Age (years),Temperature (°C),Humidity (%),Rainfall (mm),Rain Status (0/1),Soil Type,Soil Type (Numeric),Coconut Count
1930-05-31,25.233333333333334,31.333333333333332,41.9,5,27.266666666666666,67.43333333333334,5.025,1,Red Yellow Podzolic,4,511.0
1930-06-30,25.233333333333334,31.333333333333332,41.9,4,27.266666666666666,67.43333333333334,5.025,1,Red Yellow Podzolic,4,511.0
1930-07-31,30.433333333333337,31.433333333333334,46.166666666666664,5,28.133333333333336,65.86666666666667,2.6958333333333333,0,Red Yellow Podzolic,4,483.0"""
        
        with open(historical_data_path, 'w') as f:
            f.write(sample_data)
        
        print(f"Created sample historical data file at {historical_data_path}")
    
    # Use threaded=True for better concurrency
    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)