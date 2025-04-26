import requests
import matplotlib.pyplot as plt

# Запрос к Overpass API (дороги в Нижневартовске)
overpass_url = "https://overpass-api.de/api/interpreter"
query = """
[out:json];
area[name="Нижневартовск"]->.a;
(
  way["highway"](area.a);
);
out geom;
"""
response = requests.get(overpass_url, params={"data": query})
data = response.json()

# Создаём график с чёрным фоном
plt.figure(figsize=(10, 10), facecolor='black')
ax = plt.axes()
ax.set_facecolor('black')

# Рисуем дороги белым
for way in data["elements"]:
    if "geometry" in way:
        lons = [node["lon"] for node in way["geometry"]]
        lats = [node["lat"] for node in way["geometry"]]
        plt.plot(lons, lats, color='white', linewidth=0.1)

# Убираем оси
plt.axis('off')
plt.savefig("nizhnevartovsk_roads_test3.png", bbox_inches='tight', pad_inches=0, dpi=2000)
plt.close()
print("Готово: nizhnevartovsk_roads_test3.png")