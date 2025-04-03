import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

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
