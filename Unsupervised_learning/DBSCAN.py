import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("cleaned_dataset.csv")  # Adjust path if needed

# Drop unnecessary columns if they exist
df = df.drop(columns=[col for col in ['nan', 'Disease'] if col in df.columns], errors='ignore')

# Standardize the data
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

# Apply DBSCAN with balanced parameters
dbscan = DBSCAN(eps=0.8, min_samples=15)
labels = dbscan.fit_predict(df_scaled)

# Add cluster labels to the DataFrame
df['Cluster'] = labels

# Evaluate the clustering
num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
noise = list(labels).count(-1)
silhouette = silhouette_score(df_scaled, labels) if num_clusters > 1 else "N/A"

# Print the results
print(f"Number of clusters: {num_clusters}")
print(f"Noise points: {noise}")
print(f"Silhouette Score: {silhouette}")
print(df['Cluster'].value_counts())

# Save the result to a CSV file
df.to_csv("dbscan_final_result.csv", index=False)

# Step: Visualize DBSCAN Clusters using PCA
pca = PCA(n_components=2)
data_2d = pca.fit_transform(df_scaled)

# Assign colors to clusters - gray for noise (-1)
colors = ['gray' if label == -1 else f'C{label}' for label in labels]

plt.figure(figsize=(8, 6))
plt.scatter(data_2d[:, 0], data_2d[:, 1], c=colors, s=50)
plt.title('DBSCAN Clustering Visualization (eps=0.8, min_samples=15)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.grid(True)
plt.show()
