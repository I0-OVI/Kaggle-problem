# Filtering process

After downloading and displaying the csv file, we can finally start to observe the pattern of individual file.

The data seems from relational database. Most of available data is stored in the file called **sales_train.csv**. Before we start to process these csv files, filtering some abnormal items is necessary. First, I wrote a program (actually I asked AI to write a program) aiming to search the maximum and minimum value as well as the rows with negative values in **item_cnt_day** column. 
```python
input_file = 'sales_train.csv'  
output_file = 'filter_sales_train.csv'

max_price = df['item_price'].max()
min_price = df['item_price'].min()
print(f"item_price maximum value: {max_price}")
print(f"item_price minimum value: {min_price}")

negative_cnt_rows = df[df['item_cnt_day'] < 0]
negative_cnt_rows['row_number'] = negative_cnt_rows.index + 1

negative_cnt_rows.to_csv(output_file, index=False, columns=['row_number', 'date', 'date_block_num', 'shop_id', 'item_id', 'item_price', 'item_cnt_day'])
```
I decided to remove the anomalous(the negative values) because it was few impact to the final prediction due to the large amount of data which had the daily precision recordings.

However, this decision will lead to a very big issue when predicting the monthly quantity for each item in each shop where the details are put in following part.

The filter_sales_train.csv exceeds the recommended maximum size and does not be uploaded.

Back to [Readme.md](/README.md)
