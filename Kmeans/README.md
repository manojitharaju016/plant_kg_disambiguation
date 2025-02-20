# KG disambiguation
## Multisecting K-Means Clustering with FAISS CPU

This repository contains a Python script that performs multisecting K-Means clustering on node embeddings using the FAISS CPU version. The clustering process iteratively splits clusters until all clusters contain no more than a specified maximum number of points. During execution, the progress is logged to [Weights & Biases (W&B)](https://wandb.ai/), and the final clustering results are saved to a Parquet file.

## Repository Files

- **`cpufaiss_clustering_kmeans.py`**: Script for CPU K-means clustering in a hierarchical fashion
- **`hierarchical_kmeans_gpu.py`**: Script for GPU K-means clustering in a hierarchical fashion
- **`minibatch_kmeans.py`**: Script fpr CPU mini batch K-means clustering
- **`single_kmeans.py`**: Script for full CPU K-means clustering



- **`all_merged_node_embeddings.parquet`**: Input file containing the node embeddings (generated using openAI text-embedding-ada-002) and their corresponding IDs and definitions.


## Anaconda Environment Setup

This project requires Python 3.8+ along with the following packages:
- [numpy](https://numpy.org/)
- [pandas](https://pandas.pydata.org/)
- [pyarrow](https://arrow.apache.org/docs/python/) (for Parquet file I/O)
- [faiss-cpu](https://github.com/facebookresearch/faiss) (CPU version of FAISS)
- [Weights & Biases (wandb)](https://wandb.ai/) (for logging)
- [tqdm](https://github.com/tqdm/tqdm) (optional, for progress visualization)
- [scikit-learn](https://scikit-learn.org/)
- [scikit-learn-intelex](https://github.com/intel/scikit-learn-intelex) (for multi-threaded optimizations)
-  A suite of GPU-accelerated machine learning libraries (part of RAPIDS) that includes clustering algorithms like KMeans.  
  [https://rapids.ai/](https://rapids.ai/)

You can set up the environment using Anaconda with the following commands:

```bash
# Create a new conda environment named 'kmeans_env'
conda create -n kmeans_env python=3.8 -y

# Activate the environment
conda activate kmeans_env

# Install packages from conda
conda install numpy pandas pyarrow tqdm -y

# Install FAISS (CPU version) and wandb via pip
pip install faiss-cpu wandb
```
## Anaconda environment setup for GPU

This script leverages GPU library, cuML for efficient KMeans clustering. Please ensure you have a CUDA-compatible GPU and the appropriate CUDA toolkit installed. The following instructions assume you are using CUDA 11.2 (adjust accordingly for your setup).

```bash
# 1. Create and Activate a New Conda Environment
conda create -n kmeans_gpu_env python=3.8 -y
conda activate kmeans_gpu_env

# Install packages from conda
conda install numpy pandas pyarrow -y
pip install wandb
pip install cupy-cuda112

# Install cuML
conda install -c rapidsai -c nvidia -c conda-forge cuml cudatoolkit=11.2 -y
```

## MiniBatch-KMeans Clustering with scikit-learn and sklearnex

This repository contains a script that performs K-Means clustering on node embeddings using scikit-learn's `MiniBatchKMeans`, enhanced with Intel's scikit-learn extensions (`sklearnex`) for multi-threading performance. The script logs progress to Weights & Biases (W&B) and saves the final clusters in a Parquet file.

```bash
# Create anaconda environment and activate it
conda create -n kmeans_minibatch_env python=3.8 -y
conda activate kmeans_minibatch_env

# Install required packages
conda install numpy pandas pyarrow tqdm -y
pip install wandb scikit-learn scikit-learn-intelex
```
## Single-Run K-Means Clustering with FAISS (CPU)

This repository contains a Python script that performs standard K-Means clustering on node embeddings using FAISS (CPU version). The script loads embeddings from a Parquet file, logs progress and metrics to [Weights & Biases (W&B)](https://wandb.ai/), and saves the final clusters into a Parquet file.

## Required Packages

```bash
# Create and Activate a New Conda Environment
conda create -n kmeans_faiss_env python=3.8 -y
conda activate kmeans_faiss_env

# Install required packages
conda install numpy pandas pyarrow tqdm -y
pip install faiss-cpu wandb
```
