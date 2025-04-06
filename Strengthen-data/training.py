import pandas as pd
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from torch.utils.data import DataLoader, Dataset


# Custom Dataset class
class AddressDataset(Dataset):
    def __init__(self, addresses, latitudes, longitudes, tokenizer, max_length=64):
        self.addresses = addresses
        self.latitudes = latitudes
        self.longitudes = longitudes
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.addresses)

    def __getitem__(self, idx):
        address = self.addresses[idx]
        encoding = self.tokenizer(
            address,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'latitude': torch.tensor(self.latitudes[idx], dtype=torch.float),
            'longitude': torch.tensor(self.longitudes[idx], dtype=torch.float)
        }


# Neural Network Model Definition
class GeoLocationModel(nn.Module):
    def __init__(self, embedding_size, hidden_size=128):
        super(GeoLocationModel, self).__init__()
        self.fc1 = nn.Linear(embedding_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 2)  # Output: latitude and longitude

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


# Main function
def main():
    # Read training data
    try:
        df = pd.read_csv("filtered_real_cities_with_coords.csv", encoding="utf-8")
    except UnicodeDecodeError:
        print("Encoding issue detected, trying with errors='replace'")
        df = pd.read_csv("filtered_real_cities_with_coords.csv", encoding="utf-8", errors="replace")

    # Check column names
    print("CSV file columns:", df.columns)

    # Assuming column names are 'City', 'Country', 'Latitude', 'Longitude'
    addresses = []
    latitudes = []
    longitudes = []
    for index, row in df.iterrows():
        try:
            # Concatenate address
            address = f"{row['City']}, {row['Country']}"
            # Ensure coordinates are valid floats
            lat = float(row['Latitude'])
            lon = float(row['Longitude'])
            addresses.append(address)
            latitudes.append(lat)
            longitudes.append(lon)
        except (ValueError, KeyError) as e:
            print(f"Skipping row {index + 1} due to error: {e}")
            continue

    # Load BERT model and tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    bert_model = BertModel.from_pretrained('bert-base-multilingual-cased')

    # Create dataset
    dataset = AddressDataset(addresses, latitudes, longitudes, tokenizer)
    train_loader = DataLoader(dataset, batch_size=8, shuffle=True)

    # Initialize model
    model = GeoLocationModel(embedding_size=768)  # BERT output dimension is 768
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Train model
    num_epochs = 5
    for epoch in range(num_epochs):
        for batch in train_loader:
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            latitudes = batch['latitude']
            longitudes = batch['longitude']

            # Get BERT embeddings
            with torch.no_grad():
                outputs = bert_model(input_ids=input_ids, attention_mask=attention_mask)
                embeddings = outputs.last_hidden_state[:, 0, :]  # Take [CLS] token embedding

            # Predict coordinates
            preds = model(embeddings)
            loss = criterion(preds, torch.stack([latitudes, longitudes], dim=1))

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")

    # Save model
    torch.save(model.state_dict(), 'geo_location_model.pth')
    print("Model training completed and saved!")


if __name__ == "__main__":
    main()