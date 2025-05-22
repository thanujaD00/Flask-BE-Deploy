from flask import Flask, request, jsonify
import pandas as pd
from datetime import datetime
import os
import json
import pickle
import numpy as np
import joblib
import time

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
        # Get soil type for one-hot encoding
        soil_type = soil_data['soil_type']
        
        # Initialize soil type columns to 0
        soil_type_columns = {
            'Soil_Red Yellow Podzolic': 0,
            'Soil_Lateritic': 0, 
            'Soil_Cinnamon Sand': 0,
            'Soil_Alluvial': 0,
            'Soil_Sandy Loam': 0
        }
        
        # Set the appropriate column to 1 based on soil type
        if isinstance(soil_type, str):
            column_name = f'Soil_{soil_type}'
            if column_name in soil_type_columns:
                soil_type_columns[column_name] = 1
            else:
                print(f"Warning: Unknown soil type '{soil_type}'. Using Red Yellow Podzolic as default.")
                soil_type_columns['Soil_Red Yellow Podzolic'] = 1
        elif isinstance(soil_type, (int, float)):
            # Convert numeric soil type to string name for one-hot encoding
            soil_type_map_reverse = {
                0: 'Red Yellow Podzolic',
                1: 'Lateritic',
                2: 'Cinnamon Sand',
                3: 'Alluvial',
                4: 'Sandy Loam'
            }
            soil_name = soil_type_map_reverse.get(int(soil_type), 'Red Yellow Podzolic')
            column_name = f'Soil_{soil_name}'
            soil_type_columns[column_name] = 1
        
        # Prepare agronomical features with one-hot encoded soil types
        features_dict = {
            'Temperature (°C)': float(weather_data['Temperature (°C)']),
            'Humidity (%)': float(weather_data['Humidity (%)']),
            'Rainfall (mm)': float(weather_data['Rainfall (mm)']),
            'Plant Age (years)': float(soil_data['age']),
            'Year': prediction_date.year,
            'Month': prediction_date.month
        }
        
        # Add soil type columns
        features_dict.update(soil_type_columns)
        
        # Create DataFrame and ensure correct order of columns
        expected_columns = ['Plant Age (years)', 'Temperature (°C)', 'Humidity (%)', 
                           'Rainfall (mm)', 'Rain Status (0/1)', 'Year', 'Month', 
                           'Soil_Cinnamon Sand', 'Soil_Lateritic', 'Soil_Red Yellow Podzolic', 
                           'Soil_Sandy Loam', 'Soil_Alluvial']
        
        # Create DataFrame with all columns
        agro_features = pd.DataFrame([features_dict])
        
        # Add any missing columns as zeros
        for col in expected_columns:
            if col not in agro_features.columns:
                agro_features[col] = 0
                
        # Ensure we only include columns the model was trained with
        model_columns = agro_model.get_booster().feature_names
        if model_columns:
            # Select only columns the model knows about
            agro_features = agro_features[model_columns]
        
        # Print for debugging
        print(f"Feature columns: {agro_features.columns.tolist()}")
        print(f"Feature types: {agro_features.dtypes}")
        
        # Get agronomical prediction
        agro_prediction = agro_model.predict(agro_features)[0]
        
        # For time series prediction, create features based on date
        # CHANGED: Fix the time series prediction - convert to proper format based on model type
        try:
            # Check model type to determine correct input format
            if hasattr(ts_model, 'named_steps') and 'regressor' in ts_model.named_steps:
                # Pipeline with preprocessing
                ts_features = pd.DataFrame({
                    'year': [prediction_date.year],
                    'month': [prediction_date.month],
                    'day': [prediction_date.day]
                })
                ts_prediction = ts_model.predict(ts_features)[0]
            else:
                # Direct timestamp input or simple model
                # Try different formats until one works
                try:
                    # Try with timestamp
                    ts_prediction = ts_model.predict(prediction_date)[0]
                except:
                    try:
                        # Try with date string
                        ts_prediction = ts_model.predict(prediction_date.strftime('%Y-%m-%d'))[0]
                    except:
                        # Fall back to features DataFrame
                        ts_features = pd.DataFrame({
                            'ds': [prediction_date],  # Prophet uses 'ds' for dates
                            'year': [prediction_date.year],
                            'month': [prediction_date.month],
                            'day': [prediction_date.day]
                        })
                        ts_prediction = ts_model.predict(ts_features)[0]
        except Exception as ts_error:
            print(f"Time series prediction error: {ts_error}")
            # Fallback to using agronomical prediction if time series fails
            ts_prediction = agro_prediction
        
        # Combine predictions for ensemble
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

# @lru_cache(maxsize=128)
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
                
                # Map values to the expected format
                month_name = prediction_date.strftime('%B')
                
                # Format the prediction with the field names your app expects
                formatted_prediction = {
                    'month': month,
                    'month_name': month_name,
                    'year': prediction_year,
                    'ensemble_prediction': round(prediction['ensemble_prediction'], 2),
                    'long_term_prediction': round(prediction['agronomical_prediction'], 2),
                    'seasonal_prediction': round(prediction['time_series_prediction'], 2),
                    'confidence_score': round(70 + (30 * np.random.random()), 2),  # Example confidence score
                    'input_data': {
                        'plant_age': soil_data['age'],
                        'soil_type': soil_data['soil_type'],
                        'temperature': weather_data['Temperature (°C)'],
                        'humidity': weather_data['Humidity (%)'],
                        'rainfall': weather_data['Rainfall (mm)'],
                    }
                }
                
                all_predictions.append(formatted_prediction)
                
            except Exception as e:
                return jsonify({
                    'status': 'error',
                    'message': f'Error processing month {month}: {str(e)}'
                }), 400

        if all_predictions:
            # Calculate the average prediction from ensemble predictions
            average_prediction = round(
                sum(p['ensemble_prediction'] for p in all_predictions) / len(all_predictions), 
                2
            )
            
            print(f"Prediction completed in {time.time() - start_time:.2f} seconds")
            return jsonify({
                'status': 'success',
                'year': prediction_year,
                'monthly_predictions': all_predictions,
                'average_prediction': average_prediction
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

    
# Use threaded=True for better concurrency
app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)