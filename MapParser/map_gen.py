import pandas as pd
import folium

# 1. Загрузка данных о водителях и заказах
drivers_df = pd.read_csv('Data/drivers.csv')  # Водители
orders_df = pd.read_csv('Data/orders.csv')    # Заказы

# 2. Проверка данных (опционально)
print("Данные водителей:")
print(drivers_df.head())
print("\nДанные заказов:")
print(orders_df.head())

# 3. Создание карты с центром в средних координатах
# Используем координаты из обоих DataFrame для вычисления центра
all_latitudes = pd.concat([drivers_df['locationlatitude'], 
                          orders_df['fromlatitude'], 
                          orders_df['tolatitude']])
all_longitudes = pd.concat([drivers_df['locationlongitude'], 
                           orders_df['fromlongitude'], 
                           orders_df['tolongitude']])

if not all_latitudes.empty and not all_longitudes.empty:
    map_center = [all_latitudes.mean(), all_longitudes.mean()]
    drivers_map = folium.Map(location=map_center, zoom_start=12, 
                            tiles="https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}", 
                            attr="Google Maps")
    
    # 4. Добавляем водителей на карту (синие точки)
    for _, driver in drivers_df.iterrows():
        folium.CircleMarker(
            location=[driver['locationlatitude'], driver['locationlongitude']],
            radius=5,
            color='blue',
            fill=True,
            fill_color='blue',
            fill_opacity=1.0,
            popup=f"Driver ID: {driver['driver_id']}"
        ).add_to(drivers_map)
    
    # 5. Добавляем точки заказов:
    # - Красные: начальные точки (from)
    for _, order in orders_df.iterrows():
        folium.CircleMarker(
            location=[order['fromlatitude'], order['fromlongitude']],
            radius=5,
            color='red',
            fill=True,
            fill_color='red',
            fill_opacity=1.0,
            popup=f"Order ID: {order['order_id']} (From)"
        ).add_to(drivers_map)
    
    # - Зеленые: конечные точки (to)
    for _, order in orders_df.iterrows():
        folium.CircleMarker(
            location=[order['tolatitude'], order['tolongitude']],
            radius=5,
            color='green',
            fill=True,
            fill_color='green',
            fill_opacity=1.0,
            popup=f"Order ID: {order['order_id']} (To)"
        ).add_to(drivers_map)
    
    # 6. Сохраняем карту
    drivers_map.save('drivers_orders_map.html')
    print("Карта сохранена в файл 'drivers_orders_map.html'. Откройте его в браузере.")
else:
    print("Нет данных для создания карты.")