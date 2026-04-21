from typing import Union, Tuple
import numpy as np
import scipy.sparse as sp
from config import pd

class CarDataset:

    def __init__(self, X: Union[np.ndarray, sp.spmatrix, pd.DataFrame],
                 y: Union[np.ndarray, list, pd.Series]):

        self.X = self._convert_to_array(X)
        self.y = self._convert_to_array(y).squeeze()

        if len(self.X) != len(self.y):
            raise ValueError(f"特征和标签数量不匹配: X有{len(self.X)}条，y有{len(self.y)}条")

    @staticmethod
    def _convert_to_array(data) -> np.ndarray:

        if hasattr(data, "toarray"):
            return data.toarray()
        elif hasattr(data, "values"):
            return data.values
        else:
            return np.asarray(data)

    def get_data(self) -> Tuple[np.ndarray, np.ndarray]:

        return self.X, self.y

    def get_shape(self) -> Tuple[int, int]:

        return self.X.shape


def get_data_splits(
        X_train,
        X_test,
        y_train,
        y_test
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:

    train_dataset = CarDataset(X_train, y_train)
    test_dataset = CarDataset(X_test, y_test)

    return train_dataset.get_data(), test_dataset.get_data()

def get_numpy_arrays(
        X_train,
        X_test,
        y_train,
        y_test
) -> dict:

    train_X, train_y = CarDataset(X_train, y_train).get_data()
    test_X, test_y = CarDataset(X_test, y_test).get_data()

    return {
        'X_train': train_X,
        'X_test': test_X,
        'y_train': train_y,
        'y_test': test_y
    }
