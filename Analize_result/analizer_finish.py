import pandas as pd
import matplotlib.pyplot as plt


orders_df = pd.read_csv('./Data/driver_order_result.csv')
drivers_df = pd.read_csv('./Data/drivers.csv')

orders_count = orders_df['driver_id'].value_counts().reset_index()
orders_count.columns = ['driver_id', 'order_count']


drivers_orders = pd.merge(drivers_df, orders_count, on='driver_id', how='left')
drivers_orders['order_count'] = drivers_orders['order_count'].fillna(0).astype(int)


def categorize_drivers(count):
    if count == 0:
        return 'Не задействованные'
    elif count == 1:
        return 'Задействованные один раз'
    else:
        return 'Задействованные несколько раз'

drivers_orders['category'] = drivers_orders['order_count'].apply(categorize_drivers)


category_order = [
    'Задействованные один раз',
    'Задействованные несколько раз',
    'Не задействованные'
]


category_counts = drivers_orders['category'].value_counts()[category_order]

colors = ['#66b3ff', '#99ff99', '#ff9999']


plt.figure(figsize=(10, 8))
wedges, texts, autotexts = plt.pie(
    category_counts,
    labels=None,
    autopct='%1.1f%%',
    startangle=90,
    colors=colors,
    pctdistance=0.85,
    wedgeprops={'edgecolor': 'white', 'linewidth': 0.5},
    textprops={'fontsize': 12}
)


plt.legend(
    wedges,
    category_counts.index,
    title="Категории водителей",
    loc="upper right",
    bbox_to_anchor=(1, 1),
    fontsize=10
)

plt.axis('equal')
plt.title('Распределение таксистов по активности', pad=20, fontsize=14)
plt.tight_layout()
plt.show()