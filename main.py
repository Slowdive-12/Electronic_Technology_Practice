from config import set_seed
from load_data import load_data
from preprocess import process_data
from dataset import get_loader
from model import Model, train_model
from visualization import create_visualization_gui

if __name__ == "__main__":
    set_seed()
    print("加载数据...")
    df = load_data()

    print("\n数据预处理...")
    X_train, X_test, y_train, y_test, preprocessor = process_data(df)
    train_loader, test_loader = get_loader(X_train, X_test, y_train, y_test)

    print("\n 开始训练...")
    model = Model(X_train.shape[1])
    loss_history = train_model(model, train_loader, epochs=20)

    print("\n 启动主界面...")
    create_visualization_gui(df, loss_history, model, preprocessor)