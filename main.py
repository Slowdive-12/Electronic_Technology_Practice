from config import set_seed, r2_score#随机种子和R²评估函数
from load_data import load_data#用于读取CSV数据
from preprocess import process_data#特征和数据处理
from model import Model#随机森林模型类
from visualization import run_visualization#数据可视化

# 1. 画图
run_visualization("used_cars.csv")

# 2. 数据处理
set_seed()
df = load_data()
X_train, X_test, y_train, y_test = process_data(df)

# 3. 构建并训练模型
print("\n开始训练...")
model = Model(n_estimators=100, max_depth=15, random_state=42)
model.fit(X_train, y_train)
print("训练完成！")

# 4. 模型评估
print("\n模型评估...")
mse, r2, y_pred = model.evaluate(X_test, y_test)
print(f"测试集 MSE: {mse:,.2f}")
print(f"测试集 R²: {r2:.4f}")

# 5. 特征重要性
print("\n特征重要性 Top5:")
importance = model.get_feature_importance()
top5_idx = importance.argsort()[-5:][::-1]
for idx in top5_idx:
    print(f"  特征 {idx}: {importance[idx]:.4f}")

# 6. 保存模型
model.save_model("car_model.pkl")
print("\n训练完成！")
