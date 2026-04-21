from config import pd
#加载并清洗二手车数据的函数    
def load_data():
    path = "used_cars.csv"
    # 功能：读取原始二车CSV数据，完成数据清洗、过滤与特征计算
    df = pd.read_csv(path)

    # ========== 关键：清洗列，转成数字，关取CSV文件 ==========
    # 1. 清洗price列：去掉$和逗号，转成浮点数
    df["price"] = df["price"].replace(r'[\$,]', '', regex=True)
    df["price"] = pd.to_numeric(df["price"], errors="coerce")  # 转失败的变成NaN

    # 2. 清洗milage列：去掉逗号、空格和mi.，转成浮点数
    df["milage"] = df["milage"].replace(r'[,\smi.]', '', regex=True)
    df["milage"] = pd.to_numeric(df["milage"], errors="coerce")

    # ========== 过滤无效数据 ==========
    # 先删掉转换失败的NaN行，再做比较
    df = df.dropna(subset=["price", "milage", "model_year"])
    df = df[(df["price"] > 0) & (df["milage"] > 0)]

    # 计算车龄
    df["car_age"] = 2026 - df["model_year"]
    df = df[df["car_age"] <= 30]

    print(f" 数据加载完成，共{len(df)}条有效记录")
    return df
