import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import json
import os

def analyze_seasonal_patterns(historical_data_path):
    """Analyze seasonal patterns from historical data"""
    try:
        # Read the dataset
        df = pd.read_csv(historical_data_path)
        
        # Convert date string to datetime
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Extract month and year
        df['month'] = df['Date'].dt.month
        df['year'] = df['Date'].dt.year
        
        # Use 'Coconut Count' as the yield value
        monthly_stats = df.groupby('month')['Coconut Count'].agg([
            'mean', 'std', 'min', 'max', 'count'
        ]).round(2)
        
        overall_mean = df['Coconut Count'].mean()
        monthly_stats['seasonal_factor'] = (monthly_stats['mean'] / overall_mean).round(2)
        
        # Save seasonal factors to a file
        seasonal_factors = {str(k): v for k, v in monthly_stats['seasonal_factor'].to_dict().items()}
        os.makedirs('models', exist_ok=True)
        with open('models/seasonal_factors.json', 'w') as f:
            json.dump(seasonal_factors, f)
        
        # Calculate yearly variations if multiple years exist
        if len(df['year'].unique()) > 1:
            yearly_variation = df.pivot_table(
                index='month',
                columns='year',
                values='Coconut Count',
                aggfunc='mean'
            ).round(2)
            yearly_var_dict = yearly_variation.to_dict()
        else:
            yearly_var_dict = {"note": "Not enough years in the dataset for year-over-year comparison"}
        
        return {
            'monthly_stats': monthly_stats.to_dict('index'),
            'seasonal_factors': seasonal_factors,
            'yearly_variation': yearly_var_dict,
            'overall_mean': round(overall_mean, 2)
        }
        
    except Exception as e:
        print(f"Error analyzing seasonal patterns: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_seasonality_plot(historical_data_path):
    """Create visualization of seasonal patterns"""
    try:
        df = pd.read_csv(historical_data_path)
        df['Date'] = pd.to_datetime(df['Date'])
        df['month'] = df['Date'].dt.month
        
        plt.figure(figsize=(12, 6))
        
        # Calculate monthly averages
        monthly_avg = df.groupby('month')['Coconut Count'].mean()
        
        # Create the plot
        plt.plot(monthly_avg.index, monthly_avg.values, marker='o', linewidth=2)
        
        # Add month labels
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        plt.xticks(range(1, 13), month_names)
        
        # Add title and labels
        plt.title('Average Monthly Coconut Yield', fontsize=16)
        plt.xlabel('Month', fontsize=14)
        plt.ylabel('Average Coconut Count', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        # Annotate values
        for i, val in enumerate(monthly_avg.values):
            plt.annotate(f'{val:.1f}', 
                         (monthly_avg.index[i], val), 
                         textcoords="offset points",
                         xytext=(0,10), 
                         ha='center')
        
        # Save plot to memory
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100)
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        
        return base64.b64encode(image_png).decode('utf-8')
        
    except Exception as e:
        print(f"Error creating seasonality plot: {e}")
        import traceback
        traceback.print_exc()
        return None