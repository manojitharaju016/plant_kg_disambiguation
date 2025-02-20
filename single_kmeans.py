from tqdm import tqdm
import numpy as np
import pandas as pd
import faiss  # FAISS CPU version
import wandb

# Load from the compressed Parquet file
df_all_embeddings = pd.read_parquet(
    'all_merged_node_embeddings.parquet',
    engine='pyarrow'
)

# Convert Series of lists to a 2D NumPy array of shape (num_rows, 1536)
embeddings = np.array(df_all_embeddings['embedding'].tolist(), dtype='float32')

# Verify shape
print("Embeddings shape:", embeddings.shape)  # Should be (N, 1536)

wandb.init(
    project="kmeans_clustering",  # Replace with your project name
    name="kmeans_single_run_cpu",   # Customize the run name
)

def single_kmeans(
    embeddings, 
    df_all_embeddings,
    k=10,
    max_iter=100,
    random_state=42
):
    """
    Perform a standard K-Means on CPU (using FAISS) for a given k.
    Logs basic info (like cluster count, iteration step) to Weights & Biases.
    """

    # ---------------------------------------------------------------------
    # (1) Ensure embeddings are on CPU (numpy float32)
    # ---------------------------------------------------------------------
    if not isinstance(embeddings, np.ndarray):
        embeddings = np.asarray(embeddings)
    embeddings = embeddings.astype(np.float32)  # FAISS typically expects float32
    
    # Set the number of threads to use (optional, remove if you rely on defaults)
    faiss.omp_set_num_threads(32)

    # ---------------------------------------------------------------------
    # (2) Logging setup
    # ---------------------------------------------------------------------
    iteration_step = 0
    wandb.log({
        "iteration_step": iteration_step,
        "num_points": embeddings.shape[0],
        "k": k
    })
    iteration_step += 1

    # ---------------------------------------------------------------------
    # (3) Single-run K-Means
    # ---------------------------------------------------------------------
    d = embeddings.shape[1]  # dimensionality of embeddings
    kmeans = faiss.Kmeans(
        d=d,
        k=k,
        niter=max_iter,
        seed=random_state,
        verbose=False
    )
    # Train
    print("Training K-Means...")
    kmeans.train(embeddings)
    # Assign points to nearest centroid
    print("Assigning points to nearest centroid...")
    distances, pred_labels = kmeans.index.search(embeddings, 1)
    labels = pred_labels.reshape(-1)

    # ---------------------------------------------------------------------
    # (4) Group results into final DataFrame
    # ---------------------------------------------------------------------
    rows = []
    # We can either join the labels to df_all_embeddings or build a separate table
    print("Building final DataFrame...")
    for cluster_id in range(k):
        # Find indices where the assigned cluster is cluster_id
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

# Example usage:
df_sizes = single_kmeans(
     embeddings=embeddings,
     df_all_embeddings=df_all_embeddings,
     k=15000,        # specify the desired number of clusters
     max_iter=100,
     random_state=1
)
wandb.finish()

print('Saving the final clusters...')
df_sizes.to_parquet('kmeans_clusters_15k.parquet', engine='pyarrow', compression='zstd')
