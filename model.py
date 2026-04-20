import torch
import torch.nn as nn
import torch.optim as optim

class Model(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)

def train_model(model, train_loader, epochs=20, lr=0.001):
    criterion = nn.MSELoss()
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_history = []

    print("\n开始训练...")
    for epoch in range(epochs):
        loss_sum = 0
        for x, y in train_loader:
            pred = model(x)
            loss = criterion(pred, y.unsqueeze(1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_sum += loss.item()

        avg_loss = loss_sum / len(train_loader)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch + 1} Loss: {avg_loss:.2f}")

    print("训练完成！")
    return loss_history