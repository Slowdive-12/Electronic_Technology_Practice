import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk, messagebox

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


def predict_car_price(model, preprocessor, input_df):
    x = preprocessor.transform(input_df)
    pred_log = model.predict(x)
    return round(np.exp(pred_log.item()), 2)


def show_predict_result_plot(pred_price, input_info, df):
    brand = input_info['brand']
    milage = input_info['milage']
    car_age = input_info['car_age']

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.bar(['预测价格'], [pred_price], color='#4285F4', width=0.6)
    plt.title('二手车价格预测结果')
    plt.ylabel('价格（美元）')
    plt.grid(alpha=0.3)
    plt.text(0, pred_price + 2000, f'${pred_price:,.0f}', ha='center', fontsize=13, weight='bold')

    plt.subplot(2, 2, 2)
    brand_mean = df[df['brand'] == brand]['price'].mean()
    plt.bar(['预测价格', f'{brand}均价'], [pred_price, brand_mean], color=['#4285F4', '#EA4335'])
    plt.title('同品牌均价对比')

    plt.subplot(2, 2, 3)
    ages = np.arange(1, 21)
    plt.plot(ages, np.exp(-ages / 8) * pred_price * 2.5, 'o-', color='#34A853')
    plt.axvline(car_age, color='red', linestyle='--', label=f'车龄：{car_age}年')
    plt.title('车龄影响趋势')
    plt.legend()

    plt.subplot(2, 2, 4)
    milages = np.linspace(10000, 200000, 20)
    plt.plot(milages, np.exp(-milages / 80000) * pred_price * 1.8, 's-', color='#FBBC05')
    plt.axvline(milage, color='red', linestyle='--', label=f'里程：{milage:.0f}')
    plt.title('里程影响趋势')
    plt.legend()

    plt.suptitle(f'✅ {brand} 预测价格：${pred_price:,.0f}', fontsize=16, weight='bold')
    plt.tight_layout()
    plt.show()


def show_predict_window(model, preprocessor, df):
    window = tk.Tk()
    window.title("二手车价格预测系统")
    window.geometry("600x500")

    tk.Label(window, text="二手车价格预测", font=("微软雅黑", 18)).pack(pady=10)

    tk.Label(window, text="品牌：").pack()
    brand_cb = ttk.Combobox(window, values=sorted(df['brand'].dropna().unique()))
    brand_cb.pack()
    brand_cb.current(0)

    tk.Label(window, text="车型：").pack()
    model_cb = ttk.Combobox(window, values=sorted(df['model'].dropna().unique())[:60])
    model_cb.pack()
    model_cb.current(0)

    tk.Label(window, text="里程数（英里）：").pack()
    milage_entry = tk.Entry(window)
    milage_entry.pack()
    milage_entry.insert(0, "50000")

    tk.Label(window, text="车龄（年）：").pack()
    age_entry = tk.Entry(window)
    age_entry.pack()
    age_entry.insert(0, "5")

    tk.Label(window, text="燃油类型：").pack()
    fuel_cb = ttk.Combobox(window, values=sorted(df['fuel_type'].dropna().unique()))
    fuel_cb.pack()
    fuel_cb.current(0)

    tk.Label(window, text="变速箱：").pack()
    trans_cb = ttk.Combobox(window, values=sorted(df['transmission'].dropna().unique()))
    trans_cb.pack()
    trans_cb.current(0)

    def on_predict():
        try:
            input_data = pd.DataFrame({
                'brand': [brand_cb.get()],
                'model': [model_cb.get()],
                'milage': [float(milage_entry.get())],
                'car_age': [int(age_entry.get())],
                'fuel_type': [fuel_cb.get()],
                'transmission': [trans_cb.get()]
            })
            price = predict_car_price(model, preprocessor, input_data)
            info = {'brand': brand_cb.get(), 'milage': float(milage_entry.get()), 'car_age': int(age_entry.get())}
            show_predict_result_plot(price, info, df)
        except:
            messagebox.showerror("错误", "输入格式不正确！")

    tk.Button(window, text="开始预测", command=on_predict, bg="#4285F4", fg="white", font=("微软雅黑", 12)).pack(
        pady=20)
    window.mainloop()