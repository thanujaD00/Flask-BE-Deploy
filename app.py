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

app = Flask(__name__)

with open('models/coconut_price_model.pkl', 'rb') as file:
    model_data = pickle.load(file)

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
                
                # Extract soil and weather data
                soil_data = {
                    'sm_10': month_data.get('sm_10'),
                    'sm_20': month_data.get('sm_20'),
                    'sm_30': month_data.get('sm_30'),
                    'age': month_data.get('age'),
                    'soil_type': month_data.get('soil_type')
                }
                
                weather_data = {
                    'Temperature (°C)': month_data.get('Temperature (°C)'),
                    'Humidity (%)': month_data.get('Humidity (%)'),
                    'Rainfall (mm)': month_data.get('Rainfall (mm)'),
                    'Weather Description': month_data.get('Weather Description', 'normal')
                }

                # Validate input data
                for key, value in {**soil_data, **weather_data}.items():
                    if value is None:
                        return jsonify({
                            'status': 'error',
                            'message': f'Missing parameter for month {month}: {key}'
                        }), 400
                
                # Create prediction date
                prediction_date = pd.Timestamp(year=prediction_year, month=month, day=15)
                
                # Make prediction
                prediction = ensemble_predict(soil_data, weather_data, prediction_date=prediction_date)
                if prediction:
                    all_predictions.append(prediction)
            except Exception as e:
                return jsonify({
                    'status': 'error',
                    'message': f'Error processing month {month}: {str(e)}'
                }), 400

        if all_predictions:
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
    
    app.run(debug=True, host='0.0.0.0', port=5000)