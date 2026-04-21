from sklearn.ensemble import RandomForestRegressor#随机森林模型
from sklearn.metrics import mean_squared_error, r2_score#评估
import joblib#保存和加载
#参数
class Model:
    def __init__(self, n_estimators=100, max_depth=None, random_state=42):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,#树
            max_depth=max_depth,#深度
            random_state=random_state#随机数种子
        )
    #模型训练，数据规律的学习
    def fit(self, x_train, y_train):
        self.model.fit(x_train, y_train)
        return self
    #预测新数据
    def predict(self, x):
        return self.model.predict(x)
    #评估
    def evaluate(self, x_test, y_test):
        y_pred = self.model.predict(x_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        return mse, r2, y_pred
    #特征
    def get_feature_importance(self):
        return self.model.feature_importances_
    #保存模型
    def save_model(self, path):
        joblib.dump(self.model, path)
        print(f"模型已保存至: {path}")
    #加载模型
    def load_model(self, path):
        self.model = joblib.load(path)
        print(f"模型已加载自: {path}")
