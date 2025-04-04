import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Step 1: Load the dataset
df = pd.read_csv("cleaned_dataset.csv")  # Update path if needed

# Step 2: Remove unnecessary columns
df = df.dropna(axis=1, how='all')  # Remove any empty columns

if 'Disease' in df.columns:
    df = df.drop(columns=['Disease'])  # Remove class label (target)

# Step 3: Standardize Data (ONLY if not binary)
if not df.isin([0, 1]).all().all():
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)  # Convert to NumPy array
else:
    df_scaled = df.values  # Use raw values if binary

# Step 4: Use the Elbow Method to find optimal K
inertia = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    kmeans.fit(df_scaled)
    inertia.append(kmeans.inertia_)

# Step 5: Plot Elbow Method to visualize optimal K
plt.plot(K_range, inertia, marker='o')
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method for Optimal K")
plt.show()

# Step 6: Choose optimal K (adjust this after looking at the plot)
k_optimal = 5  # Replace with the observed elbow point

# Step 7: Apply K-Means with the chosen K
kmeans = KMeans(n_clusters=k_optimal, random_state=42, n_init='auto')
df['Cluster'] = kmeans.fit_predict(df_scaled)

# Step 8: Save clustered data
df.to_csv("K-mean_clustered_diseases.csv", index=False)

# Step 9: Check Silhouette Score for validation
sil_score = silhouette_score(df_scaled, df['Cluster'])
print(f"Silhouette Score: {sil_score:.4f}")

# Step 10: Print results
print(df.head())
print(df['Cluster'].value_counts())  # Check cluster distribution
# Step 11: Visualize Clusters using PCA
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
data_2d = pca.fit_transform(df_scaled)

plt.figure(figsize=(8, 6))
plt.scatter(data_2d[:, 0], data_2d[:, 1], c=df['Cluster'], cmap='Set2', s=50)
plt.title(f'K-Means Clustering Visualization (k={k_optimal})')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.colorbar(label='Cluster')
plt.grid(True)
plt.show()
