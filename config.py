# 所有库统一导入
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# 机器学习
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# 绘图中文设置
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 随机种子
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)