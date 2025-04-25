# Plant_kg_resolution

This repository provides the source code and associated resources for entity as well as relationship resolution pipeline specifically designed for plant-centric knowledge graphs. The core objective is to systematically identify and resolve ambiguous entity and relationship references within the graph. This is achieved through a multi-stage process involving vector embedding generation, clustering technique, and LLM-based finetuing and inference, ultimately enhancing the knowledge graph's accuracy and internal consistency.

## Requirements

- **Python 3.9+**
- **OpenAI** (for embedding generation)
- **tqdm** (for progress bars)
- **tiktoken** (for token counting within the OpenAI model)
- **An active OpenAI API key** (to authenticate requests)
- **csv** (built-in Python library)  
- **json** (built-in Python library)
- **pandas** (for data manipulation)
  URL: [https://pandas.pydata.org/](https://pandas.pydata.org/)
- **pyarrow**  
  URL: [https://arrow.apache.org/docs/python/](https://arrow.apache.org/docs/python/)
- **jsonlines**  
  URL: [https://pypi.org/project/jsonlines/](https://pypi.org/project/jsonlines/)
---

## Conda Environment Setup

Below are steps to create and activate a conda environment. We will install some packages using `conda` and the rest (like `openai` and `tiktoken`) using `pip`, since they may not be available on default conda channels.

1. **Create a new environment** (e.g., `embed`):
   ```bash
   conda create -n kg_resolution python=3.9 pandas tqdm
   ```
2.	**Install required packages**
   ```bash
    pip install openai tiktoken openai jsonlines
   ```

## Entity resolution
This [directory](entity_resolution/) contains scripts for entity resolution in the plant knowledge graph disambiguation pipeline. These scripts help identify and merge duplicate entities.
## Files Overview

-   **[generate_entity_embeddings.py](entity_resolution/generate_entity_embeddings.py):** Creates vector embeddings for entities along with its types and definitions in the knowledge graph. Transforms entity text/attributes into numerical representations for similarity comparison.
-   **[cpufaiss_kmeans_entities.py](entity_resolution/cpufaiss_kmeans_entities.py):** Performs K-means clustering on entity embeddings using the CPU-based FAISS library.
    Groups potentially similar entities based on their vector representations.
-   **[o3mini_inference.py](entity_resolution/o3mini_inference.py):** Runs inference using the baseline o3-mini model for finegrained sub-clustering.
-   **[validate_with_o3mini.ipynb](entity_resolution/validate_with_o3mini.ipynb):** Validates entity resolution sub-clustering results using the same o3-mini model but with a different set of instructions.
-   **[prepare_finetuning_data.ipynb](entity_resolution/prepare_finetuning_data.ipynb):** Prepares training data for 4o-mini model using o3-mini validated results; here the o3-mini corrected subclusters used as the ground-truth.
-   **[finetuned_4omini_inference.ipynb](entity_resolution/finetuned_4omini_inference.ipynb):** Runs inference with a fine-tuned 4o-mini model to match the o3-mini performance in the sub-clustering task.
-   **[collapse_entities_in_kg.ipynb](entity_resolution/collapse_entities_in_kg.ipynb):** Merges duplicate entities in the knowledge graph based on resolution results.
    Applies the final entity resolution decisions to create a consolidated knowledge graph.

### Dependencies

-   Python 3.7+
-   pandas
-   numpy
-   FAISS
-   PyTorch
-   Jupyter Notebook

### Usage

See individual script documentation for specific usage instructions.



## Type resolution



## Relationship resolution





