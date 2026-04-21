import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk
from predict import show_predict_window

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

def create_full_dashboard(df, loss_history, model, preprocessor):

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('二手车价格预测分析系统', fontsize=16, weight='bold')

    # ------------------- 1. 模型训练损失曲线 -------------------
    if loss_history and len(loss_history) > 0:
        axes[0,0].plot(range(1, len(loss_history)+1), loss_history, 'b-o')
    axes[0,0].set_title('模型训练损失曲线')
    axes[0,0].set_xlabel('训练轮次(Epoch)')
    axes[0,0].set_ylabel('损失值(MSE)')
    axes[0,0].grid(alpha=0.3)

    # ------------------- 2. 车龄分布 -------------------
    axes[0,1].hist(df['car_age'], bins=range(0, 32), color='#AED6F1', edgecolor='black')
    axes[0,1].set_title('二手车车龄分布')
    axes[0,1].set_xlabel('车龄(年)')
    axes[0,1].set_ylabel('车辆数量')
    axes[0,1].grid(axis='y', alpha=0.3)

    # ------------------- 3. 里程数分布 -------------------
    axes[0,2].hist(df['milage'], bins=30, color='#ABEBC6', edgecolor='black')
    axes[0,2].set_title('二手车里程数分布')
    axes[0,2].set_xlabel('里程数(英里)')
    axes[0,2].set_ylabel('车辆数量')
    axes[0,2].grid(axis='y', alpha=0.3)

    # ------------------- 4. 价格分布 -------------------
    axes[1,0].hist(df['price'], bins=30, color='#F5B7B1', edgecolor='black')
    axes[1,0].set_title('二手车价格分布')
    axes[1,0].set_xlabel('价格(美元)')
    axes[1,0].set_ylabel('车辆数量')
    axes[1,0].grid(axis='y', alpha=0.3)

    # ------------------- 5. TOP10品牌车辆数量 -------------------
    top_brands = df['brand'].value_counts().head(10)
    axes[1,1].bar(top_brands.index, top_brands.values, color='#F8C471')
    axes[1,1].set_title('TOP10品牌车辆数量')
    axes[1,1].tick_params(axis='x', rotation=45)
    axes[1,1].grid(axis='y', alpha=0.3)

    # ------------------- 6. 车龄与价格相关性 -------------------
    axes[1,2].scatter(df['car_age'], df['price'], color='#BB8FCE', alpha=0.6)
    axes[1,2].set_title('车龄与价格相关性')
    axes[1,2].set_xlabel('车龄(年)')
    axes[1,2].set_ylabel('价格(美元)')
    axes[1,2].grid(alpha=0.3)

    plt.tight_layout(rect=[0, 0.1, 1, 0.95])  # 给按钮留底部空间

    # 创建按钮（
    from matplotlib.widgets import Button
    ax_button = plt.axes([0.4, 0.03, 0.2, 0.05])
    btn = Button(ax_button, '打开价格预测器', color='#4285F4', hovercolor='#70A1FF')

    def open_predictor(event):
        plt.close()  # 关闭仪表盘，打开预测窗口
        show_predict_window(model, preprocessor, df)

    btn.on_clicked(open_predictor)
    plt.show()

# 兼容原接口
def create_visualization_gui(df, loss_history, model, preprocessor):
    create_full_dashboard(df, loss_history, model, preprocessor)