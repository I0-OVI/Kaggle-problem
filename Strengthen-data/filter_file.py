import pandas as pd
import os
    # Set working directory
new_path = "C:\\Users\\zhang\\Desktop\\kaggle_project"
os.chdir(new_path)
import chardet

# Detect file encoding
with open("real_cities_with_coords.csv", "rb") as f:
    result = chardet.detect(f.read())
print(f"Detected file encoding: {result['encoding']}")

# Define a function to check if string contains only English characters
def is_english(text):
    try:
        # Try encoding to ASCII - if fails, contains non-English characters
        text.encode('ascii')
        return True
    except UnicodeEncodeError:
        return False

# Read and process file line by line
data = []
with open("real_cities_with_coords.csv", "r", encoding=result['encoding'], errors="replace") as f:
    for line in f:
        try:
            # Try decoding and splitting the line
            parts = line.strip().split(",")
            if len(parts) >= 4:  # Ensure line has enough columns
                city = parts[0]
                country = parts[1]
                latitude = float(parts[2])
                longitude = float(parts[3])
                # Check if city name is in English
                if is_english(city):
                    data.append({
                        "City": city,
                        "Country": country,
                        "Latitude": latitude,
                        "Longitude": longitude
                    })
        except (UnicodeDecodeError, ValueError, IndexError) as e:
            print(f"Skipping unparseable line: {line.strip()}, error: {e}")
            continue

# Save as CSV file
df = pd.DataFrame(data)
df.to_csv("filtered_real_cities_with_coords.csv", index=False, encoding="utf-8")

print(f"Generated {len(data)} filtered addresses, saved to filtered_real_cities_with_coords.csv")