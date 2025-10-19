import pandas as pd
import matplotlib.pyplot as plt
from shapely import wkt
from shapely.geometry import Polygon

df = pd.read_csv("sample.csv")
df["geom"] = df["geom"].apply(wkt.loads)

fig, ax = plt.subplots(figsize=(10, 10))
ax.set_aspect("equal")
ax.axis("off")

colors = {
    "ROOM": "#cce5ff",
    "BATHROOM": "#b3d9ff",
    "KITCHEN": "#ffd699",
    "LIVING_ROOM": "#ffcc99",
    "BALCONY": "#e6ffe6",
    "CORRIDOR": "#f2f2f2",
    "WALL": "#000000",
    "DOOR": "#8b0000",
    "WINDOW": "#0099cc",
    "ENTRANCE_DOOR": "#ff3333",
}

for _, row in df.iterrows():
    geom = row["geom"]
    if not isinstance(geom, Polygon):
        continue
    color = colors.get(row["entity_subtype"], "#dddddd")
    x, y = geom.exterior.xy
    ax.fill(x, y, color=color, edgecolor="black", linewidth=0.5)

plt.tight_layout()
plt.savefig("floorplan.png", dpi=300, bbox_inches="tight")
plt.close()
