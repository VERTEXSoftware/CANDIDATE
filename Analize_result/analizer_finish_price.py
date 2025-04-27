import pandas as pd
import matplotlib.pyplot as plt


orders_df = pd.read_csv('./Data/driver_order_result.csv')

orders_df = orders_df[orders_df['ride_price'] > 0]


plt.figure(figsize=(10, 6))
plt.hist(
    orders_df['ride_price'],
    bins=30,
    color='green',
    alpha=0.7
)

plt.xlabel('Стоимость поездки (руб)', fontsize=12)
plt.ylabel('Количество заказов', fontsize=12)
plt.title('Распределение стоимости заказов', fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()