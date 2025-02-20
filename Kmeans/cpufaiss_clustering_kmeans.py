
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
    name="multisect_kmeans_sec_all_cpu_max_clust_800",   # Customize the run name
)

def multisect_kmeans(
    embeddings, 
    df_all_embeddings,
    max_cluster_size=50, 
    k_split=5,     
    max_iter=100,
    random_state=42
):
    """
    Perform a 'multisecting' K-Means on CPU (FAISS) until all clusters 
    have <= max_cluster_size points.
    Logs the maximum cluster size in the stack to Weights & Biases 
    at each iteration.
    """

    # ---------------------------------------------------------------------
    # (1) Ensure embeddings are on CPU (numpy float32)
    # ---------------------------------------------------------------------
    if not isinstance(embeddings, np.ndarray):
        embeddings = np.asarray(embeddings)
    embeddings = embeddings.astype(np.float32)  # FAISS typically expects float32
    
    # Set the number of threads to use (optional, remove if you rely on default)
    faiss.omp_set_num_threads(128)
    
    N = embeddings.shape[0]
    # We'll keep track of indices using a numpy array
    all_indices = np.arange(N, dtype=np.int64)
    stack = [(all_indices, embeddings)]
    
    rows = []
    cluster_id_counter = 0

    # We'll keep track of iteration steps for W&B
    iteration_step = 0
    LOG_EVERY = 25000  # Adjust logging frequency as needed

    # MAIN LOOP
    while stack:
        # Pop a cluster from the stack
        indices, sub_embeds = stack.pop()
        cluster_size = len(indices)

        # ---------------------------------------------------------------------
        # (A) Log the largest cluster size currently in the stack to W&B
        # ---------------------------------------------------------------------
        if stack:
            largest_stack_size = max(len(sub_i) for sub_i, _ in stack)
        else:
            largest_stack_size = 0

        if iteration_step % LOG_EVERY == 0:
            wandb.log({
                "iteration_step": iteration_step,
                "largest_cluster_size_in_stack": largest_stack_size,
                "current_cluster_size": cluster_size,
                "remaining_clusters_in_stack": len(stack)
            })
        # Log after the conditional as well
        wandb.log({
            "iteration_step": iteration_step,
            "largest_cluster_size_in_stack": largest_stack_size,
            "current_cluster_size": cluster_size,
            "remaining_clusters_in_stack": len(stack)
        })
        iteration_step += 1

        # ---------------------------------------------------------------------
        # (B) Check if this cluster is final or needs splitting
        # ---------------------------------------------------------------------
        if cluster_size <= max_cluster_size:
            # Final cluster
            node_ids = df_all_embeddings.loc[indices, 'node_id'].tolist()
            rows.append({
                "cluster_id": cluster_id_counter,
                "cluster_size": cluster_size,
                "node_ids_in_cluster": node_ids
            })
            cluster_id_counter += 1

        else:
            # We need to split into k_split sub-clusters using FAISS CPU KMeans
            d = sub_embeds.shape[1]  # dimensionality of embeddings
            kmeans = faiss.Kmeans(
                d=d,
                k=k_split,
                niter=max_iter,
                seed=random_state,
                verbose=False
            )
            # Train
            kmeans.train(sub_embeds)
            # Assign points to nearest centroid
            distances, pred_labels = kmeans.index.search(sub_embeds, 1)
            labels = pred_labels.reshape(-1)  # shape (cluster_size,)

            # Push each sub-cluster back onto the stack
            for label_val in range(k_split):
                mask = (labels == label_val)
                sub_indices = indices[mask]
                if len(sub_indices) == 0:
                    continue  # skip empty sub-clusters
                sub_embeds_new = sub_embeds[mask]
                stack.append((sub_indices, sub_embeds_new))

    # After all splits are done, we have our final clusters
    return pd.DataFrame(rows)


# Example usage:
df_sizes = multisect_kmeans(
     embeddings=embeddings,
     df_all_embeddings=df_all_embeddings,
     max_cluster_size=30,
     k_split=10,
     max_iter=100,
     random_state=42
)
wandb.finish()
print('saving the final clusters...')
df_sizes.to_parquet('multisect_kmeans_all_clusters_max_clust_300.parquet', engine='pyarrow',compression='zstd')
