import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import os

# Set working directory
new_path = "...."
os.chdir(new_path)

# 1. Data Loading and Preprocessing
# Ensure proper date column parsing
df = pd.read_csv('output_filtered.csv')
df['date'] = pd.to_datetime(df['date'], format='%d.%m.%Y')  # Explicit date format specification
print("**************")
print(df['item_id'].max)


# 2. Feature Engineering Function
def create_features(df):
    """Create time-based and shop-item features for the model"""

    # Validate date column
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        raise ValueError("Date column is not converted to datetime type")

    # Basic time features
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['quarter'] = df['date'].dt.quarter

    # Create unique shop-item identifier
    df['shop_item'] = df['shop_id'].astype(str) + "_" + df['item_id'].astype(str)

    # Encode shop-item combinations
    le = LabelEncoder()
    df['shop_item_code'] = le.fit_transform(df['shop_item'])

    # Add fixed item prices
    item_prices = df.groupby(['shop_id', 'item_id'])['item_price'].first().reset_index()
    df = df.merge(item_prices, on=['shop_id', 'item_id'], suffixes=('', '_fixed'))

    return df


# 3. Apply Feature Engineering
try:
    df = create_features(df)
except Exception as e:
    print(f"Feature engineering error: {e}")
    print("First 5 rows of raw data:")
    print(df.head())
    print("\nDate column type:", type(df['date'].iloc[0]))
    raise

# 4. Prepare Training Data
train = df[df['date'] < '2016-01-01']
features = ['year', 'month', 'quarter', 'shop_item_code', 'item_price_fixed']
target = 'item_cnt_day'

# 5. Train Random Forest Model
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(train[features], train[target])

# 6. Generate 2016 Prediction Data Structure
shop_items = df[['shop_id', 'item_id', 'shop_item', 'shop_item_code', 'item_price_fixed']].drop_duplicates()

# Create monthly date range for 2016
dates_2016 = pd.date_range('2016-01-01', '2016-12-31', freq='MS')
predict_data = pd.DataFrame({
    'date': np.tile(dates_2016, len(shop_items)),
    'shop_item_code': np.repeat(shop_items['shop_item_code'].values, len(dates_2016))
})

# Merge with shop-item information and add time features
predict_data = predict_data.merge(shop_items, on='shop_item_code')
predict_data['year'] = predict_data['date'].dt.year
predict_data['month'] = predict_data['date'].dt.month
predict_data['quarter'] = predict_data['date'].dt.quarter

# 7. Make Predictions
predict_data['item_cnt'] = model.predict(predict_data[features])

# 8. Post-processing: Ensure non-negative predictions
predict_data['item_cnt'] = predict_data['item_cnt'].clip(lower=0)

# 9. Save Results
result = predict_data[['date', 'shop_id', 'item_id', 'item_cnt']].sort_values(['shop_id', 'item_id', 'date'])
result.to_csv('2016_item_forecast.csv', index=False)

print("Prediction completed! Sample results:")
print(result.head())