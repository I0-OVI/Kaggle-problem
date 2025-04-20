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
![私有图片](https://github.com/I0-OVI/Kaggle-problem/blob/main/Static/Image/submission-1.png?raw=true)

### Combination
At first, I thought the score was too low because of the usage of many great algorithm. Once finding that many items had the average sale zero, I suspected about the filtering process which might drop the specific data of the items leading to the zero prediction. I used a testing program to locate the value of item sold by inputting the shop_id and item_id.
```python
df = pd.read_csv('output_filtered.csv')
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
What really 
