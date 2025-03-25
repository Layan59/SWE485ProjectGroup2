import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Load dataset
df = pd.read_csv("SWE485ProjectGroup2-main/cleaned_dataset.csv")

# Drop unnecessary columns
if 'nan' in df.columns:
    df = df.drop(columns=['nan'])

if 'Disease' in df.columns:
    df = df.drop(columns=['Disease'])

# Standardize the dataset
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

# Step 1: Plot Dendrogram to determine cluster count
plt.figure(figsize=(12, 6))
Z = linkage(df_scaled, method='ward')  # Ward method minimizes variance
dendrogram(Z, truncate_mode='level', p=10)  # Show first 10 cluster merges
plt.title("Dendrogram for HAC")
plt.xlabel("Data Points")
plt.ylabel("Distance")
plt.show()

# Step 2: Perform Hierarchical Clustering
num_clusters = 5  # Adjust this based on dendrogram observation
hac = AgglomerativeClustering(n_clusters=num_clusters, linkage='ward')
cluster_labels = hac.fit_predict(df_scaled)

# Assign clusters to the dataframe
df["Cluster"] = cluster_labels

# Step 3: Evaluate the Clustering
silhouette_avg = silhouette_score(df_scaled, cluster_labels)
print(f"Silhouette Score: {silhouette_avg:.4f}")

# Step 4: Display Cluster Distribution
cluster_counts = df["Cluster"].value_counts()
print("Cluster Distribution:\n", cluster_counts)

# Save the clustered data
df.to_csv("clustered_diseases_HAC.csv", index=False)

