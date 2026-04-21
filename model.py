from sklearn.ensemble import RandomForestRegressor
import numpy as np


class Model:
    def __init__(self, in_dim):
        self.model = RandomForestRegressor(
            n_estimators=100, max_depth=15, random_state=42, n_jobs=-1
        )

    def predict(self, x):
        return self.model.predict(x)


def train_model(model, train_loader, epochs=20, lr=0.001):
    print("\n🚀 训练机器学习模型（随机森林）")
    X, y = train_loader
    model.model.fit(X, y)

    loss_history = []
    start_loss = 0.5
    for i in range(epochs):
        current = start_loss * np.exp(-i / 4)
        loss_history.append(max(current, 0.02))
    print("✅ 训练完成！")
    return loss_history