import pandas as pd
import numpy as np

# Function to calculate the distance between two points using the Haversine formula
def haversine(lon1, lat1, lon2, lat2):
    R = 6371  # Earth's radius in kilometers
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

# Function to detect spikes
def is_spike(buf):
    dist_0_1 = haversine(buf[0][1], buf[0][0], buf[1][1], buf[1][0])
    dist_0_2 = haversine(buf[0][1], buf[0][0], buf[2][1], buf[2][0])
    dist_1_2 = haversine(buf[1][1], buf[1][0], buf[2][1], buf[2][0])
    return (dist_0_1 > dist_0_2) or (dist_1_2 > dist_0_2)

# Load GPX data
seaworld_before = pd.read_csv('output.csv')

# Detect spikes
spikes = []
for i in range(len(seaworld_before) - 2):
    buf = [
        (seaworld_before.iloc[i]['latitude'], seaworld_before.iloc[i]['longitude']),
        (seaworld_before.iloc[i + 1]['latitude'], seaworld_before.iloc[i + 1]['longitude']),
        (seaworld_before.iloc[i + 2]['latitude'], seaworld_before.iloc[i + 2]['longitude'])
    ]
    if is_spike(buf):
        spikes.append(seaworld_before.iloc[i + 1])

# Spike results
spikes_df = pd.DataFrame(spikes)
print(spikes_df)

# Install geopandas and shapely
import geopandas as gpd
from shapely.geometry import Point

# Create Point objects from spikes_df
geometry = [Point(xy) for xy in zip(spikes_df['longitude'], spikes_df['latitude'])]

# Create a GeoDataFrame
geo_df = gpd.GeoDataFrame(spikes_df, geometry=geometry)

# Save to .shp file
geo_df.to_file("spikes_output.shp")
