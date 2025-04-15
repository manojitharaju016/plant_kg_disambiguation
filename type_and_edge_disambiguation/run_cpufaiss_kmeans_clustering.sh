python cpufaiss_kmeans_elbow_edges.py \
    --embeddings_path /home/mads/connectome/data/embeddings/edge_embeddings/output/edge_embeddings_20250319_123044.parquet \
    --output_path /home/mads/connectome/data/embeddings/edge_embeddings/clustering \
    --max_cluster_size 30 \
    --min_k 2 \
    --max_k 21 \
    --max_iter 100 \
    --random_state 42