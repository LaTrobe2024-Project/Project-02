##전처리
from datetime import datetime
import pandas as pd

df = pd.read_csv('seaworld_before.csv')
# 문자열을 datetime 객체로 변환
for i in df['time']:
    date_str = i
    date_format = "%Y-%m-%d %H:%M:%S.%f %z"
    dt_object = datetime.strptime(date_str, date_format)

    # datetime 객체를 float로 변환 (Unix timestamp 형태)
    timestamp_float = dt_object.timestamp()
    df['time'][i]=timestamp_float

df[['date', 'time', 'timezone']] = df['time'].str.split(expand=True)
df[['hour', 'minute', 'second']] = df['time'].str.split(':', expand=True)

# 'geometry' 열을 기준으로 'POINT' 문자열 제거하고 숫자만 추출하여 새로운 열 추가
df[['longitude', 'latitude']] = df['wkb_geometry'].str.extract(r'POINT \(([-0-9.]+) ([-0-9.]+)\)').astype(float)

# Import necessary libraries
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
# Select the relevant columns from your DataFrame
X = df[['longitude', 'latitude']]
# Data Scaling (Standardization)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# Create and train a DBSCAN model
dbscan = DBSCAN(eps=0.1, min_samples=3)
df['cluster'] = dbscan.fit_predict(X_scaled)
# Create a DataFrame with relevant columns for clustering results
dbscan_df = df[['latitude', 'longitude', 'cluster']]
# Visualization of clustering results
import matplotlib.pyplot as plt
# Assign colors to clusters
colors = df['cluster'].map({-1: 'black', 0: 'green', 1: 'blue', 2: 'green'})
# Create a scatter plot
plt.scatter(df['longitude'], df['latitude'], c=colors)
plt.title('DBSCAN Clustering Result')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()


from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

X = df[['longitude', 'latitude']]

# 데이터 스케일링 (Standardization)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# KNN 모델 생성
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_scaled,df['cluster'])

# 클러스터링 결과 예측
df['cluster'] = knn_model.predict(X_scaled)

# 클러스터링 결과 시각화
plt.scatter(df['longitude'], df['latitude'], c=df['cluster'], cmap='viridis', edgecolors='k')
plt.title('KNN Clustering Result')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()