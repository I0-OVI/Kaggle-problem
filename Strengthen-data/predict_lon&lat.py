import pandas as pd
import torch
from transformers import BertTokenizer, BertModel

# Define neural network model
class GeoLocationModel(torch.nn.Module):
    def __init__(self, embedding_size, hidden_size=128):
        super(GeoLocationModel, self).__init__()
        self.fc1 = torch.nn.Linear(embedding_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, 2)  # Output latitude and longitude

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


# Load pre-trained model
def load_model(model_path):
    model = GeoLocationModel(embedding_size=768)  # BERT output dimension is 768
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set to evaluation mode
    return model


# Predict latitude and longitude
def predict_lat_lon(address, model, tokenizer, bert_model):
    # Convert address to BERT embedding
    encoding = tokenizer(address, return_tensors='pt')
    with torch.no_grad():
        outputs = bert_model(**encoding)
        embedding = outputs.last_hidden_state[:, 0, :]  # Take [CLS] token embedding
        preds = model(embedding)  # Predict coordinates
    return preds.tolist()[0]


# Main function
def main():
    # Load pre-trained model
    model = load_model('geo_location_model.pth')

    # Load BERT model and tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    bert_model = BertModel.from_pretrained('bert-base-multilingual-cased')

    # Read shop data
    df_shops = pd.read_csv("shops.csv")

    # Predict coordinates
    latitudes = []
    longitudes = []
    for index, row in df_shops.iterrows():
        shop_name = row['shop_name']  # Assuming column name is 'shop_name'
        lat, lon = predict_lat_lon(shop_name, model, tokenizer, bert_model)
        latitudes.append(lat)
        longitudes.append(lon)
        print(f"Shop: {shop_name} -> Predicted coordinates: ({lat}, {lon})")

    # Save results
    df_shops['latitude'] = latitudes
    df_shops['longitude'] = longitudes
    df_shops.to_csv("shops_with_predicted_coords.csv", index=False, encoding="utf-8")
    print("Results saved to shops_with_predicted_coords.csv")


if __name__ == "__main__":
    main()
