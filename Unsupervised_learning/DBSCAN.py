import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Load the dataset
df = pd.read_csv("cleaned_dataset.csv")  # Make sure the path is correct

# Drop unnecessary columns if they exist
df = df.drop(columns=[col for col in ['nan', 'Disease'] if col in df.columns], errors='ignore')

# Standardize the data for better clustering
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

# Apply DBSCAN with modified parameters to improve clustering
dbscan = DBSCAN(eps=1.2, min_samples=5)
labels = dbscan.fit_predict(df_scaled)

# Add cluster labels to the DataFrame
df['Cluster'] = labels

# Evaluate the clustering result
num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
noise = list(labels).count(-1)
silhouette = silhouette_score(df_scaled, labels) if num_clusters > 1 else "N/A"

# Print evaluation results
print(f"Number of clusters: {num_clusters}")
print(f"Noise points: {noise}")
print(f"Silhouette Score: {silhouette}")
print(df['Cluster'].value_counts())

# Save the result to a CSV file
df.to_csv("dbscan_tuned_result.csv", index=False)
