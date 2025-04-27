import struct
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import heapq
from typing import List, Tuple, Optional
import csv

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
    """Манхэттенское расстояние между точками a и b."""
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
                if cpbase_map[neighbor] == 0:  # Стена
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

def calculate_road_distances(cdtmap_data, orders_file, output_file="./Data/road_distances.csv"):

    h = cdtmap_data['height']
    w = cdtmap_data['width']
    base_map = np.frombuffer(cdtmap_data['data'], dtype=np.uint8)
    cpbase_map = base_map.reshape((h, w)).copy()
    

    orders = pd.read_csv(orders_file)
    

    orders['distance_fly'] = orders.apply(lambda x: Calc_fly_dist(x['fromlatitude'], x['fromlongitude'], x['tolatitude'], x['tolongitude']),axis=1 )
    

    def calc_road_distance(row, cdtmap_data, base_map):
        try:
            pos_x, pos_y = coord_to_cell(cdtmap_data, row['fromlatitude'], row['fromlongitude'])
            pos_to_x, pos_to_y = coord_to_cell(cdtmap_data, row['tolatitude'], row['tolongitude'])
            
            path_ride = astar(base_map, (pos_x, pos_y), (pos_to_x, pos_to_y))
            
            if path_ride is None:
                return row['distance_fly']
                
            return len(path_ride) * cdtmap_data['box_size_m']
        except Exception as e:
            print(f"Error calculating road distance for ride {row.get('order_id', 'unknown')}: {e}")
            return row['distance_fly']
    

    orders['distance_road'] = orders.apply(
        lambda x: calc_road_distance(x, cdtmap_data, cpbase_map),
        axis=1
    )
    

    orders[['driver_id', 'order_id', 'distance_fly', 'distance_road']].to_csv(output_file, index=False)
    print(f"Road distances saved to {output_file}")
    
    return orders

def plot_distance_distribution(orders):

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    

    ax1.hist(orders['distance_road'], bins=50, color='blue', alpha=0.7)
    ax1.set_title('Распределение расстояний по дороге')
    ax1.set_xlabel('Расстояние (метры)')
    ax1.set_ylabel('Количество поездок')
    ax1.grid(True)
    

    orders['road_fly_ratio'] = orders['distance_road'] / orders['distance_fly']
    ax2.hist(orders['road_fly_ratio'], bins=50, color='green', alpha=0.7)
    ax2.set_title('Соотношение дорога/прямая')
    ax2.set_xlabel('Отношение дорожного расстояния к расстоянию по прямой')
    ax2.set_ylabel('Количество поездок')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()
    

    print("\nСтатистика расстояний по дороге:")
    print(f"Минимальное расстояние: {orders['distance_road'].min():.2f} м")
    print(f"Среднее расстояние: {orders['distance_road'].mean():.2f} м")
    print(f"Медианное расстояние: {orders['distance_road'].median():.2f} м")
    print(f"Максимальное расстояние: {orders['distance_road'].max():.2f} м")
    
    print("\nСтатистика соотношения дорога/прямая:")
    print(f"Минимальное соотношение: {orders['road_fly_ratio'].min():.2f}")
    print(f"Среднее соотношение: {orders['road_fly_ratio'].mean():.2f}")
    print(f"Медианное соотношение: {orders['road_fly_ratio'].median():.2f}")
    print(f"Максимальное соотношение: {orders['road_fly_ratio'].max():.2f}")

if __name__ == "__main__":

    map_data = read_cdtmap("./Data/data_map.CDTMAP")
    print(f"Размер ячейки карты: {map_data['box_size_m']:.1f} метров")
    
    orders = calculate_road_distances(map_data, "./Data/driver_order_result.csv")
    

    plot_distance_distribution(orders)