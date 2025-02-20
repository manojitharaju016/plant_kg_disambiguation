import os
# Set the number of threads for OpenMP and MKL before any other imports
os.environ["OMP_NUM_THREADS"] = "32"
os.environ["MKL_NUM_THREADS"] = "32"

import numpy as np
import pandas as pd
from tqdm import tqdm
import wandb
from sklearnex import patch_sklearn
patch_sklearn()


from sklearn.cluster import MiniBatchKMeans
from sklearn.utils import shuffle

# -------------------------------------------------------------------------------------
# Load the Parquet file
# -------------------------------------------------------------------------------------
df_all_embeddings = pd.read_parquet(
    'all_merged_node_embeddings.parquet',
    engine='pyarrow'
)

# Convert embeddings to a NumPy array of shape (num_rows, 1536)
embeddings = np.array(df_all_embeddings['embedding'].tolist(), dtype='float32')
print("Embeddings shape:", embeddings.shape)  # e.g. (N, 1536)

# -------------------------------------------------------------------------------------
# Init W&B
# -------------------------------------------------------------------------------------
wandb.init(
    project="kmeans_clustering",
    name="kmeans_minibatch_multithread"
)

def single_kmeans(
    embeddings, 
    df_all_embeddings,
    k=10,
    max_iter=100,
    random_state=42,
    batch_size=1024
):
    """
    Perform K-Means clustering on CPU using scikit-learn's MiniBatchKMeans
    with multi-threading (n_jobs), logging basic info to Weights & Biases.
    """

    # 1) Ensure embeddings are float32
    if not isinstance(embeddings, np.ndarray):
        embeddings = np.asarray(embeddings)
    embeddings = embeddings.astype(np.float32)

    N, D = embeddings.shape

    # 2) Logging setup
    iteration_step = 0
    wandb.log({
        "iteration_step": iteration_step,
        "num_points": N,
        "k": k
    })
    iteration_step += 1

    # 3) Initialize MiniBatchKMeans with multi-threading
    #    n_jobs=-1 => use all available CPU cores
    print(f"Initializing MiniBatchKMeans with k={k}, max_iter={max_iter}, "
          f"batch_size={batch_size}, multi-threading...")
    kmeans = MiniBatchKMeans(
        n_clusters=k,
        random_state=random_state,
        init="k-means++",
        max_iter=1,          # We'll manually iterate with partial_fit
        batch_size=batch_size,
        verbose=0
    )

    # 4) Train in batches (partial_fit loop)
    print("Training MiniBatchKMeans (CPU) iteratively...")
    for epoch in range(max_iter):
        # Shuffle embeddings each epoch
        shuffled_embeddings = shuffle(embeddings, random_state=epoch)

        # Chunk into mini-batches
        for start_idx in range(0, N, batch_size):
            end_idx = min(start_idx + batch_size, N)
            X_batch = shuffled_embeddings[start_idx:end_idx]
            kmeans.partial_fit(X_batch)

        # Log each epoch
        wandb.log({"epoch": epoch + 1, "iteration_step": iteration_step})
        iteration_step += 1

    # 5) Assign points to nearest centroid
    print("Assigning points to nearest centroids...")
    labels = kmeans.predict(embeddings)

    # 6) Build final DataFrame exactly like original code
    print("Building final DataFrame...")
    rows = []
    for cluster_id in range(k):
        mask = (labels == cluster_id)
        cluster_indices = np.where(mask)[0]
        cluster_size = len(cluster_indices)

        # Extract node_ids for those indices
        node_ids = df_all_embeddings.loc[cluster_indices, 'node_id'].tolist()
        rows.append({
            "cluster_id": cluster_id,
            "cluster_size": cluster_size,
            "node_ids_in_cluster": node_ids
        })

    # Log final details
    wandb.log({
        "iteration_step": iteration_step,
        "final_k": k,
        "num_clusters": k
    })

    return pd.DataFrame(rows)

# -------------------------------------------------------------------------------------
# Example usage
# -------------------------------------------------------------------------------------
df_sizes = single_kmeans(
    embeddings=embeddings,
    df_all_embeddings=df_all_embeddings,
    k=15000,         # specify the desired number of clusters
    max_iter=100,
    random_state=1,
    batch_size=300000  # try bigger batch sizes if you have enough RAM
)
wandb.finish()

# Save results
print('Saving the final clusters...')
df_sizes.to_parquet('kmeans_clusters_150k_minibatch.parquet', engine='pyarrow', compression='zstd')
