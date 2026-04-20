import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ------------------- 数据可视化图 -------------------
def plot_price_distribution(df):
    plt.figure(figsize=(10,6))
    q = df['price'].quantile(0.99)
    d = df[df['price'] <= q]
    plt.hist(d['price'], bins=50, color='skyblue', edgecolor='black')
    plt.title('二手车价格分布')
    plt.xlabel('价格')
    plt.ylabel('数量')
    plt.grid(alpha=0.3)
    plt.show(block=True)

def plot_mile_vs_price(df):
    plt.figure(figsize=(10,6))
    q_p = df['price'].quantile(0.99)
    q_m = df['milage'].quantile(0.99)
    d = df[(df['price']<=q_p) & (df['milage']<=q_m)]
    plt.scatter(d['milage'], d['price'], alpha=0.5, c='orange')
    plt.title('里程 vs 价格')
    plt.xlabel('里程')
    plt.ylabel('价格')
    plt.grid(alpha=0.3)
    plt.show(block=True)

def plot_top_brands(df):
    plt.figure(figsize=(12,6))
    top = df['brand'].value_counts().head(10)
    plt.bar(top.index, top.values, color='lightgreen')
    plt.title('品牌数量 Top10')
    plt.xticks(rotation=45, ha='right')
    plt.grid(alpha=0.3)
    plt.show(block=True)

# ------------------- 训练损失曲线 -------------------
def plot_train_loss(loss_history):
    plt.figure(figsize=(10,5))
    plt.plot(range(1, len(loss_history)+1), loss_history, 'b-o')
    plt.title('训练损失曲线')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show(block=True)

# ------------------- GUI界面 -------------------
def create_visualization_gui(df, loss_history):
    root = tk.Tk()
    root.title('二手车数据分析系统')
    root.geometry('420x350')

    ttk.Label(root, text='数据可视化面板', font=('微软雅黑',14,'bold')).pack(pady=10)

    # 数据可视化按钮
    ttk.Button(root, text='价格分布', width=25,
               command=lambda: plot_price_distribution(df)).pack(pady=5)
    ttk.Button(root, text='里程-价格关系', width=25,
               command=lambda: plot_mile_vs_price(df)).pack(pady=5)
    ttk.Button(root, text='热门品牌Top10', width=25,
               command=lambda: plot_top_brands(df)).pack(pady=5)

    ttk.Separator(root, orient='horizontal').pack(fill='x', padx=30, pady=10)

    ttk.Label(root, text='模型训练结果', font=('微软雅黑',14,'bold')).pack(pady=5)

    # 只保留训练损失曲线按钮
    ttk.Button(root, text='训练损失曲线', width=25,
               command=lambda: plot_train_loss(loss_history)).pack(pady=5)

    root.mainloop()