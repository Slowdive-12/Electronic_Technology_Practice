from load_data import load_data
from preprocess import process_data
from dataset import get_loader
from model import Model, train_model
from visualization import create_visualization_gui

if __name__ == "__main__":
    print("加载数据中...")
    df = load_data()
    print(f"数据加载完成，共 {len(df)} 条记录")

    print("\n开始模型训练前处理...")
    X_train, X_test, y_train, y_test = process_data(df)
    train_loader, test_loader = get_loader(X_train, X_test, y_train, y_test)

    # 训练模型
    model = Model(X_train.shape[1])
    loss_history = train_model(model, train_loader, epochs=20, lr=0.001)

    # 启动可视化界面（只传df和loss_history）
    print("\n启动可视化界面...")
    create_visualization_gui(df, loss_history)