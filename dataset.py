from config import Dataset, DataLoader, torch

class CarDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X.toarray(), dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def get_loader(X_train, X_test, y_train, y_test, batch_size=16):
    train_dataset = CarDataset(X_train, y_train)
    test_dataset = CarDataset(X_test, y_test)
    return DataLoader(train_dataset, True, batch_size), DataLoader(test_dataset, False, batch_size)