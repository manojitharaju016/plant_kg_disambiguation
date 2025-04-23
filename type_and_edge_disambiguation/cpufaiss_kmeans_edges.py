#%%
from tqdm import tqdm


import os
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import faiss  # FAISS CPU version
import wandb



def multisect_kmeans(
    embeddings, 
    df_all_embeddings,
    max_cluster_size=30, 
    k_split=5,     
    max_iter=100,
    random_state=42
):
    """
    Perform a 'multisecting' K-Means on CPU (FAISS) until all clusters 
    have <= max_cluster_size points.
    Returns both the cluster DataFrame and total inertia.
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

    total_inertia = 0.0  # Track total inertia across all splits

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
            edge_ids = df_all_embeddings.loc[indices, 'id'].tolist()
            rows.append({
                "cluster_id": cluster_id_counter,
                "cluster_size": cluster_size,
                "ids_in_cluster": edge_ids
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
            # Train with suppressed warnings
            
            kmeans.train(sub_embeds)
            total_inertia += kmeans.obj[-1]  # Add final objective value (inertia)
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


    return pd.DataFrame(rows), total_inertia


def main():
    """Main function to run the clustering analysis."""
        

    embeddings_path = "path/to/embeddings.parquet"
    output_path = "path/to/output"
    max_cluster_size = 30
    min_k = 2
    max_k = 21
    max_iter = 100
    random_state = 42
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    


    
    # Load embeddings
    print(f"Loading embeddings from {embeddings_path}")
    df_all_embeddings = pd.read_parquet(embeddings_path)
    embeddings = np.array(df_all_embeddings['embedding'].tolist(), dtype='float32')
    print("Embeddings shape:", embeddings.shape)

    # Initialize wandb once at the start
    wandb.init(
        project="kmeans_clustering",
        name=f"multisect_kmeans_sec_all_cpu_max_clust_{max_cluster_size}",
    )

    try:
        # Elbow curve analysis
        k_values = range(min_k, max_k)

        # Load previous results if available
        results_path = os.path.join(output_path, "elbow_analysis_results.csv")
        if os.path.exists(results_path):
            results_df = pd.read_csv(results_path)
            completed_k_values = set(results_df['k_split'].tolist())
            inertias = results_df['inertia'].tolist()
            cluster_counts = results_df['num_clusters'].tolist()
        else:
            results_df = pd.DataFrame(columns=['k_split', 'inertia', 'num_clusters'])
            completed_k_values = set()
            inertias = []
            cluster_counts = []

        # Loop through k_values and continue from where it left off
        for k in tqdm(k_values, desc="Testing k_split values"):
            if k in completed_k_values:
                print(f"Skipping k_split value {k}, already computed.")
                continue
            
            print(f"Testing k_split value: {k}")
            wandb.init(
                project="kmeans_clustering",
                name=f"elbow_analysis_k_{k}",
                reinit=True
            )
            
            df_clusters, inertia = multisect_kmeans(
                embeddings=embeddings,
                df_all_embeddings=df_all_embeddings,
                max_cluster_size=max_cluster_size,
                k_split=k,
                max_iter=max_iter,
                random_state=random_state
            )
            
            inertias.append(inertia)
            cluster_counts.append(len(df_clusters))
            
            # Append new results and save immediately
            new_row = pd.DataFrame({
                'k_split': [k],
                'inertia': [inertia],
                'num_clusters': [len(df_clusters)]
            })
            results_df = pd.concat([results_df, new_row], ignore_index=True)
            results_df.to_csv(results_path, index=False)
            
            wandb.log({
                "k_split": k,
                "total_inertia": inertia,
                "num_clusters": len(df_clusters)
            })
            wandb.finish()
            
            clusters_path = os.path.join(output_path, f'edge_clusters_{k}_max_clust_{max_cluster_size}.parquet')
            df_clusters.to_parquet(clusters_path, engine='pyarrow', compression='zstd')

        # Generate final plots
        plot_results(results_df, k_values, output_path)
    
    finally:
        # Ensure wandb run is properly closed
        wandb.finish()

def plot_results(results_df, k_values, output_path):
    """Generate and save the elbow analysis plots."""
    plt.figure(figsize=(12, 6))

    # First subplot - Inertia
    plt.subplot(1, 2, 1)
    plt.plot(results_df['k_split'], results_df['inertia'], 'bo-')
    plt.xlabel('k_split')
    plt.ylabel('Total Inertia')
    plt.title('Elbow Curve - Inertia')
    plt.xticks(k_values)
    plt.grid(True)

    # Second subplot - Cluster Count
    plt.subplot(1, 2, 2)
    plt.plot(results_df['k_split'], results_df['num_clusters'], 'ro-')
    plt.xlabel('k_split')
    plt.ylabel('Number of Final Clusters')
    plt.title('Cluster Count vs k_split')
    plt.xticks(k_values)
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'elbow_analysis.png'))
    plt.close()

if __name__ == "__main__":
    main()

#%%
