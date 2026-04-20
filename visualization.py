from config import pd, plt, np, os

def run_visualization(csv_path="used_cars.csv"):
    if not os.path.exists(csv_path):
        print("找不到文件！")
        return

    df = pd.read_csv(csv_path)

    # 清洗
    df['price'] = df['price'].astype(str).str.replace('$', '').str.replace(',', '').astype(float)
    df['milage'] = df['milage'].astype(str).str.replace(' mi.', '').str.replace(',', '').astype(float)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('二手车市场大数据可视化分析', fontsize=22, fontweight='bold', color='#2c3e50')

    # 1 价格分布
    axes[0,0].hist(df['price'], bins=50, color='#3498db', edgecolor='white', alpha=0.8)
    axes[0,0].set_title('二手车售价分布', fontsize=14)
    axes[0,0].set_xlabel('价格 USD')
    axes[0,0].grid(alpha=0.3)

    # 2 品牌TOP10
    top_brands = df['brand'].value_counts().head(10)
    axes[0,1].barh(top_brands.index[::-1], top_brands.values[::-1], color=plt.cm.Paired(np.linspace(0,1,10)))
    axes[0,1].set_title('车源最多十大品牌', fontsize=14)

    # 3 里程 vs 价格
    axes[1,0].scatter(df['milage'], df['price'], alpha=0.5, c='#e67e22', s=15)
    axes[1,0].set_title('里程与价格相关性', fontsize=14)
    axes[1,0].set_xlim(0, df['milage'].quantile(0.95))

    # 4 燃油类型
    fuel = df['fuel_type'].value_counts()
    if len(fuel) > 5:
        fuel = pd.concat([fuel[:5], pd.Series([fuel[5:].sum()], index=['Other'])])
    axes[1,1].pie(fuel, labels=fuel.index, autopct='%1.1f%%', startangle=140, explode=[0.1]+[0]*(len(fuel)-1))
    axes[1,1].set_title('燃油类型占比')

    plt.tight_layout(rect=[0,0.03,1,0.95])
    plt.savefig('car_analysis_report.png', dpi=300)
    print("可视化图片已保存：car_analysis_report.png")
    plt.show()