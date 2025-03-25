import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Step 1: Load the dataset
df = pd.read_csv("SWE485ProjectGroup2-main/cleaned_dataset.csv")  # Ensure the correct path

# Step 2: Remove unnecessary columns
if 'nan' in df.columns:
    df = df.drop(columns=['nan'])  # Remove any stray "nan" column

if 'Disease' in df.columns:
    df = df.drop(columns=['Disease'])  # Remove class label (target)

# Step 3: Standardize Data
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)  # Keep as NumPy array

# Step 4: Apply DBSCAN
dbscan = DBSCAN(eps= 0.5, min_samples=10)  # Adjust 'eps' and 'min_samples' based on results
clusters = dbscan.fit_predict(df_scaled)

# Step 5: Assign Cluster Labels to DataFrame
df['Cluster'] = clusters

# Step 6: Evaluate DBSCAN Performance
num_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)  # Exclude noise (-1)
num_noise_points = list(clusters).count(-1)  # Count noise points

# Compute silhouette score only if there are more than 1 cluster (excluding noise)
if num_clusters > 1:
    silhouette = silhouette_score(df_scaled, clusters, metric='euclidean')
else:
    silhouette = "N/A (Only one cluster found)"

# Print Results
print(f"Number of clusters found (excluding noise): {num_clusters}")
print(f"Number of noise points: {num_noise_points}")
print(f"Silhouette Score: {silhouette}")
print(df['Cluster'].value_counts())  # Show cluster distribution

# Step 7: Save clustered data
df.to_csv("dbscan_clustered_diseases.csv", index=False)


