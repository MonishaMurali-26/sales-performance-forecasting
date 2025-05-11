import pandas as pd
import numpy as np
import sqlite3
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import warnings
import os

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Step 1: Generate or load sample data from 15+ sources (simulated as CSV files)
def generate_sample_data():
    if not os.path.exists("data_sources"):
        os.makedirs("data_sources")
    for i in range(1, 16):
        data = {
            'date': pd.date_range(start='2022-01-01', periods=36, freq='M'),
            'region': [f'Region_{i}'] * 36,
            'product': [f'Product_{(i % 5) + 1}'] * 36,
            'sales': np.random.randint(5000, 20000, 36),
            'deals_closed': np.random.randint(10, 50, 36),
            'inventory': np.random.randint(100, 1000, 36)
        }
        df = pd.DataFrame(data)
        df.to_csv(f"data_sources/source_{i}.csv", index=False)

# Step 2: ETL - Merge data from 15+ sources
def merge_data_sources():
    all_data = []
    for i in range(1, 16):
        df = pd.read_csv(f"data_sources/source_{i}.csv")
        all_data.append(df)
    merged_df = pd.concat(all_data, ignore_index=True)
    
    # Aggregate data by date
    aggregated_df = merged_df.groupby('date').agg({
        'sales': 'sum',
        'deals_closed': 'sum',
        'inventory': 'sum'
    }).reset_index()
    
    # Convert date to datetime
    aggregated_df['date'] = pd.to_datetime(aggregated_df['date'])
    return aggregated_df

# Step 3: Store merged data in SQLite (simulating data warehousing)
def store_in_warehouse(df):
    conn = sqlite3.connect("sales_warehouse.db")
    df.to_sql('sales_data', conn, if_exists='replace', index=False)
    conn.close()

# Step 4: Query data using SQL
def query_data():
    conn = sqlite3.connect("sales_warehouse.db")
    query = """
    SELECT date, sales, deals_closed, inventory,
           (deals_closed * 100.0 / SUM(deals_closed) OVER ()) AS deal_closure_rate
    FROM sales_data
    ORDER BY date
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    df['date'] = pd.to_datetime(df['date'])
    return df

# Step 5: Demand Prediction using ARIMA
def forecast_demand(df):
    # Prepare data for ARIMA
    sales_series = df.set_index('date')['sales']
    
    # Fit ARIMA model
    model = ARIMA(sales_series, order=(1, 1, 1))
    model_fit = model.fit()
    
    # Forecast next 3 months
    forecast = model_fit.forecast(steps=3)
    forecast_dates = pd.date_range(start=sales_series.index[-1] + pd.DateOffset(months=1), periods=3, freq='M')
    forecast_df = pd.DataFrame({'date': forecast_dates, 'forecasted_sales': forecast})
    
    # Calculate MSE for validation (using last 6 months as test)
    train = sales_series[:-6]
    test = sales_series[-6:]
    model_val = ARIMA(train, order=(1, 1, 1)).fit()
    pred = model_val.forecast(steps=6)
    mse = mean_squared_error(test, pred)
    print(f"Mean Squared Error (Validation): {mse:.2f}")
    
    return forecast_df

# Step 6: Generate final dataset for Power BI
def prepare_for_power_bi(df, forecast_df):
    # Merge historical and forecasted data
    df['type'] = 'Historical'
    forecast_df['type'] = 'Forecast'
    forecast_df = forecast_df.rename(columns={'forecasted_sales': 'sales'})
    forecast_df['deals_closed'] = np.nan
    forecast_df['inventory'] = np.nan
    forecast_df['deal_closure_rate'] = np.nan
    
    final_df = pd.concat([df, forecast_df], ignore_index=True)
    final_df = final_df[['date', 'sales', 'deals_closed', 'inventory', 'deal_closure_rate', 'type']]
    
    # Save to CSV
    final_df.to_csv('sales_data_for_power_bi.csv', index=False)
    print("Final dataset saved to 'sales_data_for_power_bi.csv' for Power BI import.")

# Main execution
if __name__ == "__main__":
    print("Starting Sales Performance & Forecasting Pipeline...")
    
    # Generate sample data if not exists
    if not os.path.exists("data_sources"):
        print("Generating sample data from 15 sources...")
        generate_sample_data()
    
    # ETL: Merge data
    print("Merging data from 15 sources...")
    merged_df = merge_data_sources()
    
    # Store in warehouse
    print("Storing merged data in SQLite warehouse...")
    store_in_warehouse(merged_df)
    
    # Query data
    print("Querying data with SQL...")
    queried_df = query_data()
    
    # Forecast demand
    print("Forecasting demand with ARIMA...")
    forecast_df = forecast_demand(queried_df)
    
    # Prepare final dataset for Power BI
    print("Preparing dataset for Power BI...")
    prepare_for_power_bi(queried_df, forecast_df)
    
    print("Pipeline completed successfully!")