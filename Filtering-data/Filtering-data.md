# Filtering process

After downloading and displaying the csv file, we can finally start to observe the pattern of individual file.

The data seems from relational database. Most of available data is stored in the file called **sales_train.csv**.  It is noticeable that some of the data in the column **item_cnt_day** has value smaller than zero. Our first task is to adjust or delete these kind of abnormal values. Removing these anomalous seemed to be the easiest. Also, It was few impact to the final prediction due to the large amount of data which had the daily precision recordings.

The filtered file exceeds the storage of GitHub and the filtering program is following

' 