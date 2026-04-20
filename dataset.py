from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np


# 自定义数据集类
class CarDataset(Dataset):
    def __init__(self, X, y):
        # 兼容稀疏矩阵和pandas DataFrame
        if hasattr(X, "toarray"):
            self.X = torch.tensor(X.toarray(), dtype=torch.float32)
        else:
            self.X = torch.tensor(np.array(X), dtype=torch.float32)

        if hasattr(y, "values"):
            self.y = torch.tensor(y.values, dtype=torch.float32)
        else:
            self.y = torch.tensor(np.array(y), dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def get_loader(X_train, X_test, y_train, y_test, batch_size=32):
    # 1. 创建数据集
    train_dataset = CarDataset(X_train, y_train)
    test_dataset = CarDataset(X_test, y_test)

    # 2. 创建DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader