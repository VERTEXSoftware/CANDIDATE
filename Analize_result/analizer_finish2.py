
import pandas as pd
import matplotlib.pyplot as plt



orders_df = pd.read_csv('./Data/driver_order_result.csv')
drivers_df = pd.read_csv('./Data/drivers.csv')

orders_count = orders_df['driver_id'].value_counts().reset_index()
orders_count.columns = ['driver_id', 'order_count']

# Объединение с данными о водителях
drivers_orders = pd.merge(drivers_df, orders_count, on='driver_id', how='left')
drivers_orders['order_count'] = drivers_orders['order_count'].fillna(0).astype(int)

# Категоризация водителей
def categorize_drivers(count):
    if count == 0:
        return 'Не задействованные'
    elif count == 1:
        return 'Задействованные один раз'
    else:
        return 'Задействованные несколько раз'

drivers_orders['category'] = drivers_orders['order_count'].apply(categorize_drivers)

# Задаем явный порядок категорий
category_order = [
    'Задействованные один раз',
    'Задействованные несколько раз',
    'Не задействованные'
]

# Подсчет количества водителей в каждой категории с заданным порядком
category_counts = drivers_orders['category'].value_counts()[category_order]

# Цвета для каждой категории (соответствуют новому порядку)
colors = ['#66b3ff', '#99ff99', '#ff9999']


plt.figure(figsize=(10, 6))
plt.hist(
    drivers_orders['order_count'],
    bins=range(0, drivers_orders['order_count'].max() + 2),  # Разбиваем по целым числам
    edgecolor='black',
    alpha=0.7,
    color='#66b3ff'
)

plt.xlabel('Количество заказов', fontsize=12)
plt.ylabel('Количество водителей', fontsize=12)
plt.title('Распределение количества заказов на водителя', pad=20, fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(range(0, drivers_orders['order_count'].max() + 1))  # Метки по целым числам
plt.tight_layout()
plt.show()