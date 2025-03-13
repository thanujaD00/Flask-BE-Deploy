import joblib
import json
import pandas as pd
from datetime import datetime
import os

def load_models():
    """Load both trained models"""
    try:
        long_term_model = joblib.load('models/best_yield_predictor_from_agro_data.pkl')
        seasonal_model = joblib.load('models/best_yield_predictor_for_seasonal_patterns.pkl')
        return long_term_model, seasonal_model
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        return None, None

def load_seasonal_factors():
    """Load seasonal factors from file"""
    try:
        with open('models/seasonal_factors.json', 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading seasonal factors: {e}")
        # Return default factors if file doesn't exist
        return {
            '1': 0.92, '2': 0.94, '3': 0.98, '4': 1.02,
            '5': 1.05, '6': 1.08, '7': 1.10, '8': 1.08,
            '9': 1.05, '10': 1.02, '11': 0.98, '12': 0.94
        }

def ensemble_predict(soil_data, weather_data=None, weights=None, prediction_date=None):
    """Make predictions using both models and combine them"""
    long_term_model, seasonal_model = load_models()
    seasonal_factors = load_seasonal_factors()
    
    if long_term_model is None or seasonal_model is None:
        return None
        
    if weights is None:
        weights = (0.6, 0.4)
    
    try:
        # Prepare input data for long-term model
        input_data = pd.DataFrame({
            'Soil Moisture (10 cm) (%)': [soil_data['sm_10']],
            'Soil Moisture (20 cm) (%)': [soil_data['sm_20']],
            'Soil Moisture (30 cm) (%)': [soil_data['sm_30']],
            'Plant Age (years)': [soil_data['age']],
            'Temperature (°C)': [weather_data['Temperature (°C)']],
            'Humidity (%)': [weather_data['Humidity (%)']],
            'Rainfall (mm)': [weather_data['Rainfall (mm)']],
            'Soil Type (Numeric)': [soil_data['soil_type']]
        })
        
        # Get prediction from long-term model
        long_term_pred = long_term_model.predict(input_data)[0]
        
        # Prepare time series data for seasonal model
        if prediction_date is None:
            prediction_date = pd.Timestamp.now()
            
        month = str(prediction_date.month)
        seasonal_factor = float(seasonal_factors.get(month, 1.0))
        
        # Get seasonal prediction
        try:
            ts_data = pd.DataFrame(input_data.iloc[0]).T
            ts_data.index = [prediction_date]
            seasonal_pred = seasonal_model.predict(ts_data)[0]
        except:
            print(f"Using seasonal factor adjustment for month {month}")
            seasonal_pred = long_term_pred * 1.3
        
        # Apply seasonal factor
        seasonal_pred = seasonal_pred * seasonal_factor
        
        # Combine predictions using weighted average
        ensemble_pred = (weights[0] * long_term_pred + weights[1] * seasonal_pred)
        
        # Calculate confidence score
        prediction_diff = abs(long_term_pred - seasonal_pred)
        max_diff = max(long_term_pred, seasonal_pred)
        confidence_score = (1 - prediction_diff/max_diff) * 100 if max_diff > 0 else 100
        
        return {
            'year': prediction_date.year,
            'month': prediction_date.month,
            'month_name': prediction_date.strftime('%B'),
            'ensemble_prediction': round(ensemble_pred, 2),
            'long_term_prediction': round(long_term_pred, 2),
            'seasonal_prediction': round(seasonal_pred, 2),
            'seasonal_factor': round(seasonal_factor, 2),
            'confidence_score': round(confidence_score, 2),
            'weights': weights,
            'input_data': {
                'soil_moisture_10cm': soil_data['sm_10'],
                'soil_moisture_20cm': soil_data['sm_20'],
                'soil_moisture_30cm': soil_data['sm_30'],
                'plant_age': soil_data['age'],
                'soil_type': soil_data['soil_type'],
                'temperature': weather_data['Temperature (°C)'],
                'humidity': weather_data['Humidity (%)'],
                'rainfall': weather_data['Rainfall (mm)'],
                'weather_description': weather_data['Weather Description']
            }
        }
        
    except Exception as e:
        print(f"Error making prediction: {str(e)}")
        import traceback
        traceback.print_exc()
        return None