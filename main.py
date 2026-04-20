from config import *
from load_data import load_data
from preprocess import process_data
from dataset import get_loader
from model import Model
from visualization import run_visualization  # 调用画图模块

# 1. 画图（你要的可视化）
run_visualization("used_cars.csv")

# 2. 训练模型
set_seed()
df = load_data()
X_train, X_test, y_train, y_test = process_data(df)
train_loader, test_loader = get_loader(X_train, X_test, y_train, y_test)

model = Model(X_train.shape[1])
criterion = nn.MSELoss()
opt = optim.Adam(model.parameters(), lr=0.001)

print("\n开始训练...")
for epoch in range(20):
    loss_sum = 0
    for x, y in train_loader:
        pred = model(x)
        loss = criterion(pred, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        loss_sum += loss.item()
    print(f"Epoch {epoch+1} Loss: {loss_sum:.2f}")