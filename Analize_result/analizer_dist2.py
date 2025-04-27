import struct
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import heapq
from typing import List, Tuple, Optional
from collections import defaultdict

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

def heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def astar(cpbase_map: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
    if not isinstance(cpbase_map, np.ndarray):
        raise TypeError("cpbase_map должен быть numpy.ndarray!")
    
    h, w = cpbase_map.shape
    neighbors = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    if (not (0 <= start[0] < h and 0 <= start[1] < w)) or (not (0 <= goal[0] < h and 0 <= goal[1] < w)):
        return None
    if cpbase_map[start] == 0 or cpbase_map[goal] == 0:
        return None

    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}

    open_set_hash = {start}

    while open_set:
        _, current = heapq.heappop(open_set)
        open_set_hash.remove(current)

        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1]

        for dh, dw in neighbors:
            neighbor = (current[0] + dh, current[1] + dw)

            if 0 <= neighbor[0] < h and 0 <= neighbor[1] < w:
                if cpbase_map[neighbor] == 0:
                    continue

                tentative_g_score = g_score[current] + 1

                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                    if neighbor not in open_set_hash:
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
                        open_set_hash.add(neighbor)

    return None

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
            'box_size_m': (width_m + height_m)/2.0,
            'data': output_buffer
        }

def calculate_driver_to_order_distances(cdtmap_data, drivers_file, orders_file, output_file="driver_to_order_distances.csv"):

    h = cdtmap_data['height']
    w = cdtmap_data['width']
    base_map = np.frombuffer(cdtmap_data['data'], dtype=np.uint8)
    cpbase_map = base_map.reshape((h, w)).copy()
    

    drivers = pd.read_csv(drivers_file)
    orders = pd.read_csv(orders_file)
    

    orders = orders.sort_values(['driver_id', 'order_id'])
    

    driver_locations = defaultdict(dict)
    

    for _, driver in drivers.iterrows():
        driver_locations[driver['driver_id']] = {
            'latitude': driver['locationlatitude'],
            'longitude': driver['locationlongitude']
        }
    
    results = []
    
    for _, order in orders.iterrows():
        driver_id = order['driver_id']
        

        current_loc = driver_locations[driver_id]
        

        distance_fly = Calc_fly_dist(
            current_loc['latitude'], current_loc['longitude'],
            order['fromlatitude'], order['fromlongitude']
        )
        

        def calc_road_distance():
            try:
                pos_x, pos_y = coord_to_cell(cdtmap_data, current_loc['latitude'], current_loc['longitude'])
                pos_to_x, pos_to_y = coord_to_cell(cdtmap_data, order['fromlatitude'], order['fromlongitude'])
                
                path = astar(cpbase_map, (pos_x, pos_y), (pos_to_x, pos_to_y))
                
                if path is None:
                    return distance_fly
                    
                return len(path) * cdtmap_data['box_size_m']
            except Exception as e:
                print(f"Error calculating road distance for driver {driver_id}: {e}")
                return distance_fly
        
        distance_road = calc_road_distance()
        
  
        results.append({
            'driver_id': driver_id,
            'order_id': order['order_id'],
            'distance_fly': distance_fly,
            'distance_road': distance_road,
            'from_lat': current_loc['latitude'],
            'from_lon': current_loc['longitude'],
            'to_lat': order['fromlatitude'],
            'to_lon': order['fromlongitude']
        })

        driver_locations[driver_id] = {
            'latitude': order['fromlatitude'],
            'longitude': order['fromlongitude']
        }
    

    result_df = pd.DataFrame(results)
    result_df.to_csv(output_file, index=False)
    print(f"Driver to order distances saved to {output_file}")
    
    return result_df

def plot_driver_to_order_distance_distribution(distances):

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    ax1.hist(distances['distance_road'], bins=50, color='orange', alpha=0.7)
    ax1.set_title('Статистика расстояний от водителя до заказчика (по дороге)')
    ax1.set_xlabel('Расстояние (метры)')
    ax1.set_ylabel('Количество поездок')
    ax1.grid(True)
    
    distances['road_fly_ratio'] = distances['distance_road'] / distances['distance_fly']
    ax2.hist(distances['road_fly_ratio'], bins=50, color='green', alpha=0.7, range=(1, 3))
    ax2.set_title('Соотношение дорога/прямая (водитель → заказ)')
    ax2.set_xlabel('Отношение дорожного расстояния к расстоянию по прямой')
    ax2.set_ylabel('Количество поездок')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Вывод статистики
    print("\nСтатистика расстояний от водителя до заказчика (по дороге):")
    print(f"Минимальное расстояние: {distances['distance_road'].min():.2f} м")
    print(f"Среднее расстояние: {distances['distance_road'].mean():.2f} м")
    print(f"Медианное расстояние: {distances['distance_road'].median():.2f} м")
    print(f"Максимальное расстояние: {distances['distance_road'].max():.2f} м")
    
    print("\nСтатистика соотношения дорога/прямая:")
    print(f"Минимальное соотношение: {distances['road_fly_ratio'].min():.2f}")
    print(f"Среднее соотношение: {distances['road_fly_ratio'].mean():.2f}")
    print(f"Медианное соотношение: {distances['road_fly_ratio'].median():.2f}")
    print(f"Максимальное соотношение: {distances['road_fly_ratio'].max():.2f}")

if __name__ == "__main__":

    map_data = read_cdtmap("./Data/data_map.CDTMAP")
    print(f"Размер ячейки карты: {map_data['box_size_m']:.1f} метров")
    

    distances = calculate_driver_to_order_distances(map_data, "./Data/drivers.csv", "./Data/driver_order_result.csv")
    

    plot_driver_to_order_distance_distribution(distances)