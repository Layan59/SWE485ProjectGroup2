import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, precision_recall_fscore_support
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from scipy.cluster.hierarchy import dendrogram, linkage
from tkinter import filedialog
from tkinter import Tk

# =================================================================
# 1. SMART FILE LOADING WITH MULTIPLE FALLBACKS
# =================================================================
def load_dataset():
    # Try default path first
    default_path = r"C:\Users\Asus\SWE485ProjectGroup2-main\Unsupervised_learning\cleaned_dataset.csv"
    alternative_paths = [
        r"cleaned_dataset.csv",
        r"../cleaned_dataset.csv",
        r"data/cleaned_dataset.csv"
    ]

    # Check if file exists in default location
    if os.path.exists(default_path):
        print(f"Found dataset at default location: {default_path}")
        return pd.read_csv(default_path)
    
    # Check alternative locations
    for path in alternative_paths:
        if os.path.exists(path):
            print(f"Found dataset at alternative location: {path}")
            return pd.read_csv(path)
    
    # If not found, let user browse
    print("Could not find dataset automatically. Please select the file manually.")
    root = Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select cleaned_dataset.csv",
        filetypes=[("CSV files", "*.csv")]
    )
    if file_path:
        return pd.read_csv(file_path)
    else:
        raise FileNotFoundError("No dataset file selected. Program terminated.")

def compute_bcubed(reference_labels, predicted_labels):
    precision, recall, _, _ = precision_recall_fscore_support(reference_labels, predicted_labels, average='binary')
    return precision, recall

try:
    df = load_dataset()
    
    # Debug: Show basic info about loaded data
    print("\nData loaded successfully!")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(df.head(2))

    # =================================================================
    # 2. DATA PREPROCESSING
    # =================================================================
    df = df.drop(columns=[col for col in ['nan', 'Disease'] if col in df.columns], errors='ignore')
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)

    # =================================================================
    # 3. DBSCAN CLUSTERING (with auto-tuning)
    # =================================================================
    print("\n" + "="*40 + "\nDBSCAN Analysis\n" + "="*40)

    # Auto-determine eps
    nn = NearestNeighbors(n_neighbors=15)
    nn.fit(X_scaled)
    distances, _ = nn.kneighbors(X_scaled)
    k_dist = np.sort(distances[:, -1])

    plt.figure(figsize=(10, 5))
    plt.plot(k_dist)
    plt.axhline(y=1.2, color='r', linestyle='--', label='Suggested eps=1.2')
    plt.title("k-NN Distance Plot (for DBSCAN eps tuning)")
    plt.ylabel("Distance to 15th neighbor")
    plt.legend()
    plt.show()

    # Run DBSCAN with recommended parameters
    dbscan = DBSCAN(eps=1.2, min_samples=15)
    db_labels = dbscan.fit_predict(X_scaled)
    print(f"DBSCAN clusters: {len(np.unique(db_labels))}")
    print(f"Noise points: {np.sum(db_labels == -1)}")
    silhouette_dbscan = silhouette_score(X_scaled, db_labels) if len(set(db_labels)) > 1 else "N/A"
    print(f"Silhouette: {silhouette_dbscan:.3f}")

    # =================================================================
    # 4. K-MEANS CLUSTERING
    # =================================================================
    print("\n" + "="*40 + "\nK-Means Analysis\n" + "="*40)

    # Elbow Method
    inertias = []
    k_range = range(2, 11)
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)

    plt.figure(figsize=(10, 5))
    plt.plot(k_range, inertias, 'bo-')
    plt.title("Elbow Method for Optimal K")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Inertia")
    plt.show()

    # Run with suggested K
    optimal_k = 5  # Change based on elbow plot
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init='auto')
    km_labels = kmeans.fit_predict(X_scaled)
    print(f"K-Means inertia: {kmeans.inertia_:.2f}")
    silhouette_kmeans = silhouette_score(X_scaled, km_labels)
    print(f"Silhouette: {silhouette_kmeans:.3f}")

    # =================================================================
    # 5. HIERARCHICAL CLUSTERING
    # =================================================================
    print("\n" + "="*40 + "\nHAC Analysis\n" + "="*40)

    # Dendrogram
    linked = linkage(X_scaled, method='ward')
    plt.figure(figsize=(12, 6))
    dendrogram(linked, truncate_mode='level', p=5)
    plt.title("Dendrogram (Ward Linkage)")
    plt.show()

    # Run clustering
    hac = AgglomerativeClustering(n_clusters=optimal_k, linkage='ward')
    hac_labels = hac.fit_predict(X_scaled)
    silhouette_hac = silhouette_score(X_scaled, hac_labels)
    print(f"HAC Silhouette: {silhouette_hac:.3f}")

    # =================================================================
    # 6. VISUALIZATION (PCA)
    # =================================================================
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    fig, ax = plt.subplots(1, 3, figsize=(20, 5))
    algorithms = [
        ("DBSCAN", db_labels),
        ("K-Means", km_labels),
        ("HAC", hac_labels)
    ]

    for i, (name, labels) in enumerate(algorithms):
        scatter = ax[i].scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', s=10)
        ax[i].set_title(f"{name} Clusters")
        plt.colorbar(scatter, ax=ax[i])
    plt.tight_layout()
    plt.show()

    # =================================================================
    # 7. RESULTS COMPARISON
    # =================================================================
    results = pd.DataFrame({
        "Algorithm": ["DBSCAN", "K-Means", "HAC"],
        "Silhouette": [
            silhouette_dbscan,
            silhouette_kmeans,
            silhouette_hac
        ],
        "Clusters": [
            len(np.unique(db_labels)) - (1 if -1 in db_labels else 0),
            optimal_k,
            optimal_k
        ],
        "Noise Points": [np.sum(db_labels == -1), 0, 0],
        "Inertia": ["N/A", f"{kmeans.inertia_:.2f}", "N/A"]
    })

    print("\n" + "="*40 + "\nFinal Comparison\n" + "="*40)
    print(results)

    # Assuming you have true labels in `true_labels` for BCubed
    # true_labels = [...]  # Load or define your ground truth labels here
    # bcubed_results = {
    #     "DBSCAN": compute_bcubed(true_labels, db_labels),
    #     "K-Means": compute_bcubed(true_labels, km_labels),
    #     "HAC": compute_bcubed(true_labels, hac_labels)
    # }
    # print("BCubed Precision and Recall:")
    # for method, metrics in bcubed_results.items():
    #     print(f"{method}: Precision = {metrics[0]:.4f}, Recall = {metrics[1]:.4f}")

except Exception as e:
    print(f"\nERROR: {str(e)}")
    print("Suggested fixes:")
    print("- Check if the dataset file exists")
    print("- Verify file path in the code")
    print("- Ensure you have required permissions")
    print("- Check file isn't open in another program")

finally:
    input("\nPress Enter to exit...")