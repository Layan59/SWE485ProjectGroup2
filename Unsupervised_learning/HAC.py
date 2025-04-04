import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("cleaned_dataset.csv")
df = df.drop(columns=[col for col in ['nan', 'Disease'] if col in df.columns], errors='ignore')

# Standardize
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

# Try more cluster values
best_score = -1
best_k = 0
best_labels = None

for k in range(2, 8): 
    hac = AgglomerativeClustering(n_clusters=k, linkage='ward')
    labels = hac.fit_predict(df_scaled)
    score = silhouette_score(df_scaled, labels)
    
    if score > best_score:
        best_score = score
        best_k = k
        best_labels = labels

# Assign best labels
df["Cluster"] = best_labels
df.to_csv("HAC_tuned_result.csv", index=False)

# Output
print(f"Best number of clusters: {best_k}")
print(f"Best Silhouette Score: {best_score:.4f}")
print("Cluster distribution:")
print(df["Cluster"].value_counts())

# Step: Visualize HAC Clusters using PCA
pca = PCA(n_components=2)
data_2d = pca.fit_transform(df_scaled)

plt.figure(figsize=(8, 6))
plt.scatter(data_2d[:, 0], data_2d[:, 1], c=best_labels, cmap='Set1', s=50)
plt.title(f'HAC Clustering Visualization (k={best_k})')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.colorbar(label='Cluster')
plt.grid(True)
plt.show()
