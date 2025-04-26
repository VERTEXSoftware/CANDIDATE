import struct
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import heapq

MAX_DIST_M = 4000

def draw_path(base_map, path_astar, width, height, path_value=2):
    path_map = np.copy(base_map).reshape(height, width)
    
    for x, y in path_astar:
        path_map[y, x] = path_value
    
    return path_map


def heuristic(a, b):
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

def astar(base_map, start, goal, width, height):
    if base_map[start[1] * width + start[0]] == 0 or base_map[goal[1] * width + goal[0]] == 0:
        return None
    
    neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    
    close_set = set()
    came_from = {}
    gscore = {start: 0}
    fscore = {start: heuristic(start, goal)}
    oheap = []
    heapq.heappush(oheap, (fscore[start], start))
    
    while oheap:
        current = heapq.heappop(oheap)[1]
        
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path
        
        close_set.add(current)
        
        for dx, dy in neighbors:
            neighbor = (current[0] + dx, current[1] + dy)
            
            if 0 <= neighbor[0] < width and 0 <= neighbor[1] < height:
                if base_map[neighbor[1] * width + neighbor[0]] == 0:
                    continue
                
                tentative_g_score = gscore[current] + 1
                
                if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, float('inf')):
                    continue
                
                if tentative_g_score < gscore.get(neighbor, float('inf')) or neighbor not in [i[1] for i in oheap]:
                    came_from[neighbor] = current
                    gscore[neighbor] = tentative_g_score
                    fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                    heapq.heappush(oheap, (fscore[neighbor], neighbor))  
    return None


def FIND_DRIVERS_ROUND_1_2(cdtmap_data,order,drivers):
    
    pos_lat = order['fromlatitude']
    pos_lon = order['fromlongitude']  
    pos_x, pos_y = coord_to_cell(cdtmap_data,  pos_lat, pos_lon)
    
    drivers_round1_data = []
    
    for _, driver in drivers.iterrows():
        pos_d_lat = driver['locationlatitude']
        pos_d_lon = driver['locationlongitude']
        if Calc_fly_dist(pos_lat, pos_lon, pos_d_lat,pos_d_lon)<=MAX_DIST_M:
             drivers_round1_data.append(driver.to_dict())
            
    drivers_round1 = pd.DataFrame(drivers_round1_data)
    
    
    
    return drivers

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


    start = (2320, 2252)
    goal = (2373, 2214)
    
    path_astar = astar(base_map, start, goal, w, h)
    
    fig, ax = plt.subplots(figsize=(12, 12))
    
    if path_astar:

        path_map = draw_path(base_map, path_astar, w, h)
        
        if path_map is not None:
            print("Карта с путем:")
            ax.imshow(cpbase_map, cmap='gray')
            ax.imshow(path_map, alpha=0.8)
            ax.imshow(orders_layer, alpha=0.7)
            
    else:
        print("Путь не найден!")

    

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