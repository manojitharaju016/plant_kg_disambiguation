import os
os.environ["OMP_NUM_THREADS"] = "16"
os.environ["MKL_NUM_THREADS"] = "16"

import numpy as np
import pandas as pd
import cupy as cp
import wandb
from cuml.cluster import KMeans 

# -------------------------------------------------------------------------------------
# Load data
# -------------------------------------------------------------------------------------
df_all_embeddings = pd.read_parquet(
    'all_merged_node_embeddings.parquet',
    engine='pyarrow'
)

embeddings = np.array(df_all_embeddings['embedding'].tolist(), dtype='float32')
print("Embeddings shape:", embeddings.shape)

# -------------------------------------------------------------------------------------
# Init W&B
# -------------------------------------------------------------------------------------
wandb.init(project="kmeans_clustering", name="kmeans_gpu_batchsize")

def single_kmeans_gpu(
    embeddings, 
    df_all_embeddings,
    k=10,
    max_iter=100,
    random_state=42,
    batch_size=1024
):
    """
    Perform GPU KMeans via cuML.
    """

    if not isinstance(embeddings, np.ndarray):
        embeddings = np.asarray(embeddings)
    embeddings = embeddings.astype(np.float32)
    N, D = embeddings.shape

    # Log info
    wandb.log({"num_points": N, "k": k})

    print(f"Initializing cuML KMeans with k={k}, max_iter={max_iter}, "
          f"batch_size={batch_size}...")

    # Move entire data to GPU 
    embeddings_gpu = cp.asarray(embeddings)

    # KMeans with an internal mini-batch approach
    kmeans = KMeans(
        n_clusters=k,
        max_iter=max_iter,
        random_state=random_state,
        init="k-means||",  # analogous to 'k-means++'
        max_samples_per_batch=batch_size
    )

    print("Fitting KMeans on GPU (single call)...")
    kmeans.fit(embeddings_gpu)

    print("Assigning points to nearest centroids...")
    labels_gpu = kmeans.predict(embeddings_gpu)
    labels = labels_gpu.get()  # back to CPU

    # Build final DataFrame
    print("Building final DataFrame...")
    rows = []
    for cluster_id in range(k):
        mask = (labels == cluster_id)
        cluster_indices = np.where(mask)[0]
        cluster_size = len(cluster_indices)
        node_ids = df_all_embeddings.loc[cluster_indices, 'node_id'].tolist()
        rows.append({
            "cluster_id": cluster_id,
            "cluster_size": cluster_size,
            "node_ids_in_cluster": node_ids
        })

    wandb.log({"final_k": k, "num_clusters": k})
    return pd.DataFrame(rows)

df_sizes = single_kmeans_gpu(
    embeddings=embeddings,
    df_all_embeddings=df_all_embeddings,
    k=50000,
    max_iter=100,
    random_state=1,
   batch_size=1000
)
wandb.finish()

print('Saving the final clusters...')
df_sizes.to_parquet('kmeans_clusters_50k_gpu_batchsize_1k.parquet', 
                    engine='pyarrow', 
                    compression='zstd')
