# Prediction

To reach this stage, I was a little bit hazy about the proble: I did not remember which value should I predict or calculate for. 
This was my order to finish the final part
- Longitude and Latitude prediction
- Individual item prediction
- Combination

  ### Longitude and Latitude prediction
  As I had mentioned before in the **strengthening data** part, there was a huge gap between the regions where shops located.
  This can be a good feature to predict the result.
  So, I used the **LightGBM** regression packeted with **MultiOutputRegressor**, actually recommanded by the AI.
  Here is the whole program to gain the file **2016_predictions_clean.csv**:
```python
import pandas as pd
import numpy as np
from sklearn.multioutput import MultiOutputRegressor
import lightgbm as lgb
import os

# Set working directory
new_path = "..."
os.chdir(new_path)

# Data loading and preprocessing
df = pd.read_csv('final_monthly_summary_with_coords_converted.csv', parse_dates=['date'])
df = df.sort_values(['shop_id', 'date'])

# Generate skeleton data for 2016 predictions
shops = df[['shop_id', 'longitude', 'latitude']].drop_duplicates()
dates_2016 = pd.date_range('2016-01-01', '2016-12-31', freq='MS')  # Monthly start frequency
test_2016 = pd.DataFrame({
    'shop_id': np.repeat(shops['shop_id'].values, len(dates_2016)),
    'date': np.tile(dates_2016, len(shops)),
    'longitude': np.repeat(shops['longitude'].values, len(dates_2016)),
    'latitude': np.repeat(shops['latitude'].values, len(dates_2016))
})

# Feature engineering
def add_basic_features(df):
    """Add basic time-based features"""
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['quarter'] = df['date'].dt.quarter
    return df

# Prepare training and test data
train = add_basic_features(df[df['date'] < '2016-01-01'])  # Use pre-2016 data for training
test_2016 = add_basic_features(test_2016)  # Add features to test set

# Model training configuration
features = ['shop_id', 'year', 'month', 'quarter', 'longitude', 'latitude']
targets = ['total_price', 'total_quantity']

# Initialize and train multi-output regression model
model = MultiOutputRegressor(lgb.LGBMRegressor())
model.fit(train[features], train[targets])

# Generate and save predictions
preds = model.predict(test_2016[features])
test_2016[['predicted_price', 'predicted_quantity']] = preds
test_2016[['shop_id', 'date', 'predicted_price', 'predicted_quantity']] \
    .to_csv('2016_predictions_clean.csv', index=False)
```
If you are careful enough, the input file name is final_monthly_summary_with_coords_converted.csv which had the [date conversion](\Prediction\date_conversion.py) based on the final_monthly_summary_with_coords.csv combined by the month sales and coordinates of shop.
This is because the feature engineering needs the specific format: Initially the date was like 'month.year'(eg: Jan.13) but the required format was 'year/month/day'. \
Now, this part is temporarily finished. We can turn to the next part.

### Individual item prediction
Actually, when I started to handle this section, there was confusion hold by me at that time: I literally did not know the target column.
In fact, fear occupied my heart and suspicion about the usage of previous working was thought immediately. 
I was uncertain about the avaliablity about the longitude and latitude prediction.
By dropping the precious work, only depending on the file from the first part (filtering data), a very unreliable plan was appeared in my mind.

Using data from 2013 to 2015, specifically at daily level, actually enabled me to generate a prediction whereas the amount exceeds the quality of data making the prediction reasonable to some extent.
AI selected **random forest** as the training algorithm. The program is [here](\Prediction\item_prediction.py).

Here is an intersting vignette when I submitted this extremely unreliable prediction to the Kaggle website. He told me that 'Evaluation Exception: Submission must have 214200 rows'. 
I was really confused about it, I had output all the quantities of items sold in the csv file but the total line number was much lower than the submission criteria.
Until I opened a file called **test.csv**, I realized why the line criteria was so big: The ID column in the submission file was corresponded by item_id and shop_id.  \
The score was not very high.(The score for the rank 1 person is about 0.75. I guess the score is to calculate variance bewteen values in the submitted file and the set-up file)
![](https://github.com/I0-OVI/Kaggle-problem/blob/main/Static/Image/submission-1.png?raw=true)

### Combination
At first, I thought the score was too low because of the usage of many great algorithm. Once finding that many items had the average sale zero, I suspected about the filtering process which might drop the specific data of the items leading to the zero prediction. I used a testing program to locate the value of item sold by inputting the shop_id and item_id.
```python
df = pd.read_csv('2016_item_forecast.csv')
def get_item_cnt(shop_id: int, item_id: int) -> float:
    result = df[(df['shop_id'] == shop_id) & (df['item_id'] == item_id)]
    return result['item_cnt'].values[0] if not result.empty else None
if __name__ == "__main__":
    # Query parameters input
    for i in range(10):
        query_shop_id = int(input("Enter shop_id: "))
        query_item_id = int(input("Enter item_id: "))

        # Execute query
        cnt = get_item_cnt(query_shop_id, query_item_id)

        # Output results
        print(f"\nQuery result: {cnt if cnt is not None else 'No matching records found'}")
```
What really scared me was that I searched for the values from **2016_item_forecast.csv** to **sales_train.csv** which is the raw file installed on the Kaggle website and the result was always *'No matching records found'*. Initially (I only tested two files: 2016_item_forecast.csv and output_filtered.csv), I mistakenly believed there were errors in the filtering process where the direct filtering of negative values might take responsible for this issue. But it was proved to be wrong when I searched the item in **sales_train.csv** due to *'no matching'* information returned when the same ids entered. \
Although I could plan for a new method such as prediction, I was too lazy to do this. So I decided to use the previously generated total quantity and revenue to recalibrate the final answers. First, the calibrating factor was calculated and applied to the predicted values in  **2016_item_forecast.csv**. The result was not very satisfying instead it was really bad where the values were lower than 1 and most of them were scattering around 0.11 which was unreasonable. Another method was implemented as follows. (This method performs proportional recalibration by calculating each item's contribution ratio)
```python
import pandas as pd
import os
# Set working directory
new_path = "..."
os.chdir(new_path)

# 1. Load prediction data
df_shop = pd.read_csv('2016_predictions_clean.csv', parse_dates=['date'])  # Shop-level predictions
df_item = pd.read_csv('2016_item_forecast.csv', parse_dates=['date'])     # Item-level predictions

# 2. Calculate monthly distribution ratios at item level (preserving original distribution)
df_item['item_ratio'] = df_item.groupby(['shop_id', 'date'])['item_cnt'].transform(
    lambda x: x / x.sum()
)

# 3. Merge with shop-level target totals
df_item = pd.merge(
    df_item,
    df_shop[['shop_id', 'date', 'predicted_quantity']],
    on=['shop_id', 'date'],
    how='left'
)

# 4. Distribute target quantities proportionally (key improvement)
df_item['adjusted_item_cnt'] = df_item['item_ratio'] * df_item['predicted_quantity']

# 5. Handle missing values (use 80% of original predictions as fallback)
df_item['adjusted_item_cnt'] = df_item['adjusted_item_cnt'].fillna(df_item['item_cnt'] * 0.8)

# 6. Apply minimum sales constraint (avoid over-shrinking)
df_item['adjusted_item_cnt'] = df_item['adjusted_item_cnt'].clip(lower=0.5)  # Minimum 0.5 units

# 7. Save results
df_item.to_csv('improved_item_forecast_2016.csv', index=False)

# Validation
adjusted_total = df_item.groupby(['shop_id', 'date'])['adjusted_item_cnt'].sum()
target_total = df_shop.set_index(['shop_id', 'date'])['predicted_quantity']

print("Adjusted statistics:")
print("Mean adjusted item prediction:", df_item['adjusted_item_cnt'].mean())
print("Maximum total quantity error:", (adjusted_total - target_total).abs().max())
```
Congratulations, we have completed most of the work but still need to implement the integration program:
```python
import pandas as pd
import os

# Set working directory
new_path = "C:\\Users\\zhang\\Desktop\\kaggle题目"
os.chdir(new_path)

# 1. Load data files
forecast_df = pd.read_csv('improved_item_forecast_2016.csv')  # Contains date, shop_id, item_id, item_cnt
test_df = pd.read_csv('test.csv')  # Contains ID, item_id, shop_id

# 2. Calculate monthly average sales per shop-item combination
monthly_avg = forecast_df.groupby(['shop_id', 'item_id'])['adjusted_item_cnt'].sum().reset_index()
print(monthly_avg['adjusted_item_cnt'])
monthly_avg['item_cnt_month'] = monthly_avg['adjusted_item_cnt'] / 12  # Convert annual to monthly average
print(monthly_avg)
monthly_avg.drop(columns=['adjusted_item_cnt'], inplace=True)

# 3. Merge with test data
result_df = test_df.merge(monthly_avg,
                         on=['shop_id', 'item_id'],
                         how='left')

# 4. Handle missing values (fill with 0)
result_df['item_cnt_month'] = result_df['item_cnt_month'].fillna(0)

# 5. Select only required columns for submission
final_result = result_df[['ID', 'item_cnt_month']]

# 6. Save final results
final_result.to_csv('final_result.csv', index=False)

# Validation and reporting
nan_count_before = result_df['item_cnt_month'].isna().sum()
print(f"\nProcessing summary:")
print(f"Missing values before filling: {nan_count_before}")
print(f"Missing values after filling: {result_df['item_cnt_month'].isna().sum()}")
print("\nProcessing complete! Results saved to final_result.csv")
print("Sample output:")
print(final_result.head())
```
Submit this version csv file, we got this score (huge improvement in the accuracy):
![](https://github.com/I0-OVI/Kaggle-problem/blob/main/Static/Image/submission-2.png?raw=true)
As for the prediction for the row with empty given information, emmmmmm\
**TO BE CONTINUE-->**
