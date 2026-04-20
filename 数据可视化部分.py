import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# --- 全局配置：解决中文乱码与负号显示问题 ---
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def run_visualization(csv_path):
    """
    二手车项目数据可视化核心模块
    """
    # 检查文件是否存在
    if not os.path.exists(csv_path):
        print(f"错误：未找到文件 {csv_path}，请确保该文件在项目根目录下。")
        return

    # 1. 加载数据
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"数据加载失败: {e}")
        return

    # 2. 数据清洗 (针对数据中的特殊字符)
    # 处理价格: 移除 '$' 和 ',' 并转为数值
    df['price'] = df['price'].str.replace('$', '').str.replace(',', '').astype(float)
    # 处理里程: 移除 ' mi.' 和 ',' 并转为数值
    df['milage'] = df['milage'].str.replace(' mi.', '').str.replace(',', '').astype(float)

    # 3. 创建可视化画布 (2x2 布局)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('二手车市场大数据可视化分析', fontsize=22, fontweight='bold', color='#2c3e50')

    # --- 图表 1：价格分布情况 (直方图 + 核密度估计曲线) ---
    axes[0, 0].hist(df['price'], bins=50, color='#3498db', edgecolor='white', alpha=0.8)
    axes[0, 0].set_title('二手车售价分布分布', fontsize=14)
    axes[0, 0].set_xlabel('价格 (USD)')
    axes[0, 0].set_ylabel('车辆数量')
    axes[0, 0].grid(axis='y', linestyle='--', alpha=0.3)

    # --- 图表 2：品牌车源量 Top 10 (水平条形图) ---
    top_brands = df['brand'].value_counts().head(10)
    colors = plt.cm.Paired(np.linspace(0, 1, 10))
    axes[0, 1].barh(top_brands.index[::-1], top_brands.values[::-1], color=colors)
    axes[0, 1].set_title('车源数量最多的十大品牌', fontsize=14)
    axes[0, 1].set_xlabel('挂牌数量')

    # --- 图表 3：里程与价格的相关性 (散点图) ---
    axes[1, 0].scatter(df['milage'], df['price'], alpha=0.5, c='#e67e22', s=15)
    # 添加趋势参考线
    axes[1, 0].set_title('行驶里程与售价相关性趋势', fontsize=14)
    axes[1, 0].set_xlabel('里程 (miles)')
    axes[1, 0].set_ylabel('价格 (USD)')
    axes[1, 0].set_xlim(0, df['milage'].quantile(0.95))  # 限制x轴范围，排除极端长途车干扰

    # --- 图表 4：燃油类型构成 (饼图) ---
    fuel_data = df['fuel_type'].value_counts()
    # 仅展示前5类，其余合并为Other
    if len(fuel_data) > 5:
        others = pd.Series([fuel_data[5:].sum()], index=['Other'])
        fuel_data = pd.concat([fuel_data[:5], others])

    axes[1, 1].pie(fuel_data, labels=fuel_data.index, autopct='%1.1f%%',
                   startangle=140, colors=plt.cm.Pastel1.colors, explode=[0.1] + [0] * (len(fuel_data) - 1))
    axes[1, 1].set_title('市场燃油类型占比', fontsize=14)

    # 4. 优化布局并展示
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # 自动保存一份结果到本地
    plt.savefig('car_analysis_report.png', dpi=300)
    print("可视化报表已生成并保存为 'car_analysis_report.png'")

    plt.show()


# --- 执行入口 ---
if __name__ == "__main__":
    # 请确保 used_cars.csv 文件放在 PyCharm 项目的根目录下
    DATA_FILE = 'used_cars.csv'
    run_visualization(DATA_FILE)