# Strengthening data
Finishing two previous parts, it is followed by data augmentation, which means adding more columns to the original csv in order to have a more accurate prediction.

### Contents
- Longitude and Altitude
- Monthly sales 
- 



### Longitude and Altitude
I initially questioned why this test included **shop_id.csv**, as knowing individual shop names seemed irrelevant. However, after learning about Russia's regional economic disparities, I realized shop locations could be a critical factor for our training of prediction model.

To address this, my first instinct was to integrate a **mapping API**. The approach appeared straightforward: iterate through the file and process the search results. At first glance, it seemed simple enough. Yet when I implemented this, the output CSV was filled with NaN values. Except this, it took me 10 minutes to finish the whole process due to the limited accessing time of the API. Upon reflection, I recognized the core issue: shop names alone provide insufficient information for accurate geolocation. **This clearly demands a more robust methodology.**
```python
pip install pandas geopy
```
```python
import pandas as pd 
from geopy.geocoders import Nominatim  # OpenStreetMap Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable  # abnormal process
import time  # used to control the access rate
FILE_PATH = "shops.csv"          
OUTPUT_PATH = "shops_with_coords.csv"  
def get_geocoder():
    return Nominatim(user_agent="geo_locator")
def geocode_address(address, geolocator, retries=3):
    for _ in range(retries):
        try:
            location = geolocator.geocode(address, language='ru')
            if location:
                return (location.latitude, location.longitude)
            return (None, None)
        except (GeocoderTimedOut, GeocoderUnavailable):
            time.sleep(2) 
    return (None, None)
def main():
    df = pd.read_csv(FILE_PATH)    
    geolocator = get_geocoder()
    
    df[['latitude', 'longitude']] = df['name'].apply(
        lambda x: pd.Series(geocode_address(x, geolocator)))
    df.to_csv(OUTPUT_PATH, index=False, encoding='utf-8-sig')
if __name__ == "__main__":
    main()
```
#### Embedded
**Embedded** is such a novel word to me. Actually right now I do not know the working principle of it because it uses the meaning of the context to predict the position based on the pre-trained model. First, the training dataset requires the place and its corresponding location involving the longitude and altitude. As I mentioned before, there were no returned coordinates for the given place in **shops.csv**. The generation of the position came to my mind. But I was totally deceived by the AI, suggesting me to import fake library to tackle this. I really do so, the issues occurred when I run the program searching for the individual place. It was so wired for me to see a NaN-filled output. 

A few minutes ago, I was laughing in disbelief. I should have realized immediately as I saw this library called FAKE. I told AI to have another method to replace the ridiculous library. He provided the [download link of countries](http://download.geonames.org/export/dump/cities15000.zip) and the file extraction program.
```python
df = pd.read_csv("cities15000.txt", sep="\t", header=None, encoding='utf-8')

cities = df[1].tolist()
countries = df[8].tolist()
latitudes = df[4].tolist()
longitudes = df[5].tolist()

data = {
    "City": cities,
    "Country": countries,
    "Latitude": latitudes,
    "Longitude": longitudes
}
df_output = pd.DataFrame(data)
df_output.to_csv("real_cities_with_coords.csv", index=False, encoding="utf-8")
```
We have got the country names and their coordinates which enables us to train the model (normally Word2Vec、GloVe or transformer-based). The Deepseek chose the transformer-based one using the **torch** library. The whole progress of the training is following.

1. Address embedding extraction

   Use the pre-training model(eg: BERT) to extract the semantic representation of the input address

2. Training process

   Take the extracted data as input and corresponding longtitude and altitude as output in order to get a regression model
   
3. Predict the coordinates







