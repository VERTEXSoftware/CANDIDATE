import pandas as pd
import folium

# 1. Загрузка данных
drivers_df = pd.read_csv('Data/drivers.csv')    # Водители (синие)
orders_df = pd.read_csv('Data/orders.csv')       # Заказы (красные/зеленые)
rides_df = pd.read_csv('Data/rides.csv')        # Поездки (оранжевые/желтые)

# 2. Проверка данных (опционально)
print("Данные водителей:")
print(drivers_df.head())
print("\nДанные заказов:")
print(orders_df.head())
print("\nДанные поездок:")
print(rides_df.head())

# 3. Создание карты с центром в средних координатах
# Собираем все координаты для расчета центра
all_latitudes = pd.concat([drivers_df['locationlatitude'], orders_df['fromlatitude'], orders_df['tolatitude'], rides_df['fromlatitude'], rides_df['tolatitude']])
all_longitudes = pd.concat([drivers_df['locationlongitude'],orders_df['fromlongitude'], orders_df['tolongitude'],rides_df['fromlongitude'], rides_df['tolongitude']])

if not all_latitudes.empty and not all_longitudes.empty:
    map_center = [all_latitudes.mean(), all_longitudes.mean()]
    m = folium.Map(location=map_center, zoom_start=12,
                  tiles="https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}",
                  attr="Google Maps")
    
    # 4. Добавляем водителей (синие точки)
    for _, driver in drivers_df.iterrows():
        folium.CircleMarker(
            location=[driver['locationlatitude'], driver['locationlongitude']],
            radius=5,
            color='blue',
            fill=True,
            fill_color='blue',
            fill_opacity=1.0,
            popup=f"Driver ID: {driver['driver_id']}"
        ).add_to(m)
    
    # 5. Добавляем заказы:
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
        ).add_to(m)
    
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
        ).add_to(m)
    
    # 6. Добавляем поездки:
    # - Оранжевые: начальные точки (from)
    for _, ride in rides_df.iterrows():
        folium.CircleMarker(
            location=[ride['fromlatitude'], ride['fromlongitude']],
            radius=5,
            color='orange',
            fill=True,
            fill_color='orange',
            fill_opacity=1.0,
            popup=f"Ride ID: {ride['driver_id']} (From)"
        ).add_to(m)
    
    # - Желтые: конечные точки (to)
    for _, ride in rides_df.iterrows():
        folium.CircleMarker(
            location=[ride['tolatitude'], ride['tolongitude']],
            radius=5,
            color='yellow',
            fill=True,
            fill_color='yellow',
            fill_opacity=1.0,
            popup=f"Ride ID: {ride['driver_id']} (To)"
        ).add_to(m)
    
    # 7. Добавляем легенду (объяснение цветов)
    legend_html = '''
    <div style="position: fixed; 
                bottom: 50px; left: 50px; width: 180px; height: 160px; 
                border:2px solid grey; z-index:9999; font-size:14px;
                background-color:white;
                padding: 10px;">
      <b>Цвета точек:</b><br>
      &bull; <span style="color:blue">●</span> Водители<br>
      &bull; <span style="color:red">●</span> Заказы (откуда)<br>
      &bull; <span style="color:green">●</span> Заказы (куда)<br>
      &bull; <span style="color:orange">●</span> Поездки (откуда)<br>
      &bull; <span style="color:yellow">●</span> Поездки (куда)<br>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # 8. Сохраняем карту
    m.save('full_map.html')
    print("Карта сохранена в файл 'full_map.html'. Откройте его в браузере.")
else:
    print("Нет данных для создания карты.")