import pandas as pd
from datetime import datetime


def load_data(file_path="used_cars.csv"):
    df = pd.read_csv(file_path)

    df["price"] = df["price"].astype(str).replace(r'[\$,]', '', regex=True)
    df["price"] = pd.to_numeric(df["price"], errors="coerce")

    df["milage"] = df["milage"].astype(str).replace(r'[,\smi.]', '', regex=True)
    df["milage"] = pd.to_numeric(df["milage"], errors="coerce")

    df = df.dropna(subset=["price", "milage", "model_year"])
    df = df[(df["price"] > 0) & (df["milage"] > 0)]

    df["car_age"] = datetime.now().year - df["model_year"]
    df = df[df["car_age"] <= 30]

    print(f"✅ 数据加载完成，有效数据：{len(df)} 条")
    return df