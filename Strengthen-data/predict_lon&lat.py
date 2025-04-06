import pandas as pd
import torch
from transformers import BertTokenizer, BertModel

# 定义神经网络模型
class GeoLocationModel(torch.nn.Module):
    def __init__(self, embedding_size, hidden_size=128):
        super(GeoLocationModel, self).__init__()
        self.fc1 = torch.nn.Linear(embedding_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, 2)  # 输出经纬度

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


# 加载预训练模型
def load_model(model_path):
    model = GeoLocationModel(embedding_size=768)  # BERT 输出维度为 768
    model.load_state_dict(torch.load(model_path))
    model.eval()  # 设置为评估模式
    return model


# 预测经纬度
def predict_lat_lon(address, model, tokenizer, bert_model):
    # 将地址转换为 BERT 嵌入
    encoding = tokenizer(address, return_tensors='pt')
    with torch.no_grad():
        outputs = bert_model(**encoding)
        embedding = outputs.last_hidden_state[:, 0, :]  # 取 [CLS] 位置的嵌入
        preds = model(embedding)  # 预测经纬度
    return preds.tolist()[0]


# 主函数
def main():
    # 加载预训练模型
    model = load_model('geo_location_model.pth')

    # 加载 BERT 模型和分词器
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    bert_model = BertModel.from_pretrained('bert-base-multilingual-cased')

    # 读取商店数据
    df_shops = pd.read_csv("shops.csv")

    # 预测经纬度
    latitudes = []
    longitudes = []
    for index, row in df_shops.iterrows():
        shop_name = row['shop_name']  # 假设列名为 'shop_name'
        lat, lon = predict_lat_lon(shop_name, model, tokenizer, bert_model)
        latitudes.append(lat)
        longitudes.append(lon)
        print(f"商店: {shop_name} -> 预测经纬度: ({lat}, {lon})")

    # 保存结果
    df_shops['latitude'] = latitudes
    df_shops['longitude'] = longitudes
    df_shops.to_csv("shops_with_predicted_coords.csv", index=False, encoding="utf-8")
    print("结果已保存至 shops_with_predicted_coords.csv")


if __name__ == "__main__":
    main()