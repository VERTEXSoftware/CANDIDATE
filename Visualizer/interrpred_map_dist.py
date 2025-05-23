import struct
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import heapq
from typing import List, Tuple, Optional

MAX_DIST_M = 5000

def draw_path(base_map, path_astar, width, height, path_value=200):
    path_map = np.copy(base_map).reshape(height, width)
    
    for x, y in path_astar:
        path_map[y, x] = path_value
    
    return path_map

def draw_path_on_map(cpbase_map: np.ndarray, path: List[Tuple[int, int]], path_value: int = 200) -> np.ndarray:
    """Отрисовывает путь на копии карты."""
    if path is None:
        raise ValueError("Path is None. Cannot draw.")
    path_map = cpbase_map.copy()
    for y, x in path:  # Обратите внимание: (y, x) для numpy-матрицы!
        path_map[y, x] = path_value
    return path_map

def coord_to_cell(cdtmap_data, latitude, longitude):

    south, north = cdtmap_data['x1'], cdtmap_data['x2']
    west, east = cdtmap_data['y1'], cdtmap_data['y2']

    if not (west <= longitude <= east and north <= latitude <= south):
        latitude = np.clip(latitude, north, south)
        longitude = np.clip(longitude, west, east)

    x_ratio = (longitude - west) / (east - west)
    y_ratio = (latitude - south) / (north - south)

    col = int(round(x_ratio * (cdtmap_data['width'] - 1)))
    row = int(round(y_ratio * (cdtmap_data['height'] - 1)))

    return row, col

def draw_point(orders_layer, base_map, y, x, h, w, clr):
    for i in range(-9, 9):
        for j in range(-9, 9):
            if 0 <= x+i < w and 0 <= y+j < h:
                orders_layer[y+j, x+i] = clr
                base_map[y+j, x+i] = 1



def heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    """Манхэттенское расстояние между точками a и b."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def Calc_fly_dist(lat1, lon1, lat2, lon2):
    R = 6371000.0
    
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)  
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    distance = R * c
    return distance 

def astar(cpbase_map: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
    """
    Алгоритм A* для поиска пути на матрице cpbase_map (h, w).
    - Дорога: 1, Стены: 0.
    - Возвращает список точек пути или None, если путь не найден.
    """
    if not isinstance(cpbase_map, np.ndarray):
        raise TypeError("cpbase_map должен быть numpy.ndarray!")
    
    h, w = cpbase_map.shape
    neighbors = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Вниз, вправо, вверх, влево

    # Проверка валидности старта и цели
    if (not (0 <= start[0] < h and 0 <= start[1] < w)) or (not (0 <= goal[0] < h and 0 <= goal[1] < w)):
        return None
    if cpbase_map[start] == 0 or cpbase_map[goal] == 0:
        return None

    # Инициализация
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal:
            # Восстановление пути
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1]  # Разворачиваем путь от start до goal

        for dh, dw in neighbors:
            neighbor = (current[0] + dh, current[1] + dw)

            if 0 <= neighbor[0] < h and 0 <= neighbor[1] < w:
                if cpbase_map[neighbor] == 0:  # Стена
                    continue

                tentative_g_score = g_score[current] + 1

                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return None


def FIND_DRIVERS_ROUND_1_2(cdtmap_data,base_map, order, drivers, path_map):
    
    h = cdtmap_data['height']
    w = cdtmap_data['width']
    
    pos_lat = order['fromlatitude']
    pos_lon = order['fromlongitude']  
    
    pos_to_lat = order['tolatitude']
    pos_to_lon = order['tolongitude'] 
    
    
    
    pos_x, pos_y = coord_to_cell(cdtmap_data, pos_lat, pos_lon)
    pos_to_x, pos_to_y = coord_to_cell(cdtmap_data, pos_to_lat, pos_to_lon)
    
    fly_order_dist = Calc_fly_dist(pos_lat, pos_lon, pos_to_lat, pos_to_lon)
    path_order_dist = astar(base_map, (pos_x, pos_y), (pos_to_x,pos_to_y))
    
    #draw_point(path_map, base_map,  pos_x, col_d, h, w, [64, 128, 255, 255]) 
    
    if path_order_dist == None:
        road_to_dist_m =0
    else:
        road_to_dist_m =(len(path_order_dist) - 1)*cdtmap_data['box_size_m']
        #print(road_to_dist_m)
    
    drivers_round1_data = []
    
    for _, driver in drivers.iterrows():
        pos_d_lat = driver['locationlatitude']
        pos_d_lon = driver['locationlongitude']
        fly_dist = Calc_fly_dist(pos_lat, pos_lon, pos_d_lat, pos_d_lon)
        
        if fly_dist <= MAX_DIST_M:
            
            pos_d_lat = driver['locationlatitude']
            pos_d_lon = driver['locationlongitude']
        
            pos_d_x, pos_d_y = coord_to_cell(cdtmap_data, pos_d_lat, pos_d_lon)
            
            path_drv = astar(base_map,  (pos_x, pos_y), (pos_d_x,pos_d_y))
            
            driver_data = driver.to_dict()
            driver_data['fly_dist'] = fly_dist 
            driver_data['fly_to_dist'] = fly_order_dist
            
            if path_order_dist == None:
                driver_data['road_dist'] =0
            else:
                driver_data['road_dist'] = (len(path_drv)-1)*cdtmap_data['box_size_m']
                print(driver_data['road_dist'] )
            
            driver_data['road_to_dist'] = road_to_dist_m
            drivers_round1_data.append(driver_data)
            
            if path_drv != None:
                draw_path_on_map(path_map, path_drv)
    
    
   
    
    drivers_round1_2 = pd.DataFrame(drivers_round1_data)
    
        
        
    
    return drivers_round1_2

def read_cdtmap(filename):
    with open(filename, 'rb') as file:
        x1, y1, x2, y2 = struct.unpack('4f', file.read(4 * 4))
        width, height = struct.unpack('2i', file.read(2 * 4))        
        output_buffer = file.read(width * height)
        
        lat_deg_per_cell = (x1 - x2) / height
        lon_deg_per_cell = (y2 - y1) / width
    
        avg_lat = math.radians((x1 + x2) / 2)
    
        height_m = lat_deg_per_cell * 111320
        width_m = lon_deg_per_cell * 111320 * math.cos(avg_lat)
        
        return {
            'x1': x1,
            'y1': y1,
            'x2': x2,
            'y2': y2,
            'width': width,
            'height': height,
            'box_size_m' : (width_m + height_m)/2.0,
            'data': output_buffer
        }


def plot_cdtmap_with_orders_and_drivers(cdtmap_data, orders_file, drivers_file):
    h = cdtmap_data['height']
    w = cdtmap_data['width']
    
    base_map = np.frombuffer(cdtmap_data['data'], dtype=np.uint8)
    cpbase_map = base_map.reshape((h, w)).copy()
    
    orders_layer = np.zeros((h, w, 4), dtype=np.uint8)
    
    orders = pd.read_csv(orders_file)
    drivers = pd.read_csv(drivers_file)
    
    for _, order in orders.iterrows():

        row_a, col_a = coord_to_cell(cdtmap_data, order['fromlatitude'], order['fromlongitude'])
        draw_point(orders_layer, cpbase_map, row_a, col_a, h, w, [255, 0, 0, 255])  # RGBA red
        
        row_b, col_b = coord_to_cell(cdtmap_data, order['tolatitude'], order['tolongitude'])
        draw_point(orders_layer, cpbase_map, row_b, col_b, h, w, [0, 255, 0, 255])  # RGBA blue
    
    for _, driver in drivers.iterrows():
        row_d, col_d = coord_to_cell(cdtmap_data, driver['locationlatitude'], driver['locationlongitude'])
        draw_point(orders_layer, cpbase_map, row_d, col_d, h, w, [0, 0, 255, 255])  # RGBA green

    path_map = np.zeros((h, w, 1), dtype=np.uint8)
    
    specific_order = orders[orders['order_id'] == 7828671687].iloc[0]
    
    FIND_DRIVERS_ROUND_1_2(cdtmap_data,cpbase_map,specific_order,drivers,path_map)
   
    
    fig, ax = plt.subplots(figsize=(12, 12))

    
    ax.imshow(cpbase_map, cmap='gray')
    ax.imshow(path_map, alpha=0.8)
    ax.imshow(orders_layer, alpha=0.7)

    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title(f'Orders and Drivers Map\nBounding box: {cdtmap_data["x1"]}°-{cdtmap_data["x2"]}° N, {cdtmap_data["y1"]}°-{cdtmap_data["y2"]}° E')
    
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', label='Клиент'),
        Patch(facecolor='green', label='Точка назначения'),
        Patch(facecolor='blue', label='Водители')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.show()

map_data = read_cdtmap("./Data/data_map.CDTMAP")

print(f"Размер ячейки: {map_data['box_size_m']:.1f}")
plot_cdtmap_with_orders_and_drivers(map_data, "./Data/orders.csv", "./Data/drivers.csv")