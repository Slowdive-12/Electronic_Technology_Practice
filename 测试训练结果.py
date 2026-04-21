import numpy as np
import pandas as pd
from model import Model
from config import pd
import joblib
#加载训练好的模型，开启测试
def predict_new_data(model_path="car_model.pkl"):
    from preprocess import get_preprocessor
    
    preprocessor = get_preprocessor()
    model = Model()
    model.load_model(model_path)
    
    print("\n" + "="*50)
    print("二手车价格预测系统")
    print("="*50)
    
    brand = input("请输入品牌(如Ford, BMW): ")
    model_name = input("请输入车型(如Civic, 3系): ")
    fuel = input("请输入燃料类型(如gas, diesel): ")
    trans = input("请输入变速箱(如automatic, manual): ")
    mileage = float(input("请输入里程数: "))
    year = int(input("请输入年份: "))
    
    car_age = 2026 - year
    new_data = pd.DataFrame({
        "brand": [brand],
        "model": [model_name],
        "fuel_type": [fuel],
        "transmission": [trans],
        "milage": [mileage],
        "car_age": [car_age]
    })
    
    x_new = preprocessor.transform(new_data)
    pred_price = model.predict(x_new)[0]
    
    print(f"\n预测价格: ${pred_price:,.2f}")
    print("="*50)
