import os
import openai
import pandas as pd
from openai import APIError
from time import sleep
from tqdm import tqdm
import tiktoken


#read plant knowledge graph
#read the connectome kg file
print('Loading the knowledge graph. It may take a while...')
df_kg = pd.read_csv('/path/to/your/knowledge_graph.csv')

#read the node type mapping file
mapping_df = pd.read_csv("path/to/your/label_remap_curated.csv")

df_kg["source type original"] = df_kg["source type"].copy()
df_kg["target type original"] = df_kg["target type"].copy()


df_kg["source type"] = df_kg["source type"].map(mapping_df.set_index("id")["parent_node"]).fillna("other")
df_kg["target type"] = df_kg["target type"].map(mapping_df.set_index("id")["parent_node"]).fillna("other")


df_kg = df_kg.dropna(subset=['source','source type','target','target type'])
source_nodes = df_kg[['source', 'source type','source_extracted_definition','source_generated_definition']]
target_nodes = df_kg[['target', 'target type','target_extracted_definition','target_generated_definition']]



#Merge the source and target nodes. Change the column names source_updated and target_updated to node_name and node_type
source_nodes.columns = ['name', 'node_type', 'extracted_definition','generated_definition']
target_nodes.columns = ['name', 'node_type', 'extracted_definition','generated_definition']

nodes = pd.concat([source_nodes, target_nodes])


# put id column as the first column
# generate a unique id for each node based on the node_name and node_type
nodes['id'] = nodes['name'] + ' [' + nodes['node_type'] + ']'

nodes = nodes[['id', 'name', 'node_type','extracted_definition','generated_definition']]

nodes.drop_duplicates(subset=['id'], keep='first', inplace=True)

# Function to generate embeddings for nodes
def generate_embeddings(nodes_subset, nodes_file_name, api_key, max_tokens_per_batch=8192):
    """
    Generate embeddings for nodes in the knowledge graph and save to a file.
    Args:
        nodes_subset (pd.DataFrame): DataFrame containing node data.
        nodes_file_name (str): File name to save the embeddings.
        api_key (str): OpenAI API key.
        max_tokens_per_batch (int): Maximum tokens per batch for API calls.
    Returns:
        None
    """
    openai.api_key = api_key  # Set your OpenAI API key

    print("Columns in nodes DataFrame:", nodes_subset.columns)

    node_embeddings = []
    print("Generating embeddings for nodes...")

    # Collect all input texts and metadata
    inputs = []
    metadata = []
    for _, r in tqdm(nodes_subset.iterrows(), total=nodes_subset.shape[0]):
        input_text = (
            f"{r['id']} : {r['extracted_definition']}"
        )
        inputs.append(input_text)
        metadata.append({
            'node_id': r['id'],
            'definition': r['extracted_definition'],
        })

    # Initialize tiktoken encoding for token counting
    encoding = tiktoken.encoding_for_model("text-embedding-ada-002")

    # Function to batch inputs without exceeding token limits
    def batch_inputs_by_tokens(inputs, metadata, max_tokens_per_batch):
        """
        Batch inputs based on token limits.
        Args:
            inputs (list): List of input texts.
            metadata (list): List of metadata dictionaries.
            max_tokens_per_batch (int): Maximum tokens per batch.
        Returns:
            list: List of batches of inputs.
            list: List of metadata for each batch.
        """
        batches = []
        batch_metadata = []
        current_batch = []
        current_meta = []
        current_tokens = 0

        for inp, meta in zip(inputs, metadata):
            num_tokens = len(encoding.encode(inp))
            if num_tokens > max_tokens_per_batch:
                raise ValueError(f"Input is too long: {inp}")
            if current_tokens + num_tokens > max_tokens_per_batch:
                batches.append(current_batch)
                batch_metadata.append(current_meta)
                current_batch = [inp]
                current_meta = [meta]
                current_tokens = num_tokens
            else:
                current_batch.append(inp)
                current_meta.append(meta)
                current_tokens += num_tokens

        if current_batch:
            batches.append(current_batch)
            batch_metadata.append(current_meta)
        return batches, batch_metadata

    # Batch inputs respecting token limits
    batches, metadata_batches = batch_inputs_by_tokens(inputs, metadata, max_tokens_per_batch)

    for batch_inputs, batch_meta in tqdm(zip(batches, metadata_batches), total=len(batches)):
        successful_call = False
        while not successful_call:
            try:
                # Function-based API call
                node_emb = openai.embeddings.create(
                    model="text-embedding-ada-002",
                    input=batch_inputs
                )
                successful_call = True
            except openai.OpenAIError as e:
                print(e)
                print("Retrying for node embeddings in 5 seconds...")
                sleep(5)

        # Collect embeddings and associate them with metadata
        for emb_data, meta in zip(node_emb.data, batch_meta):
            embedding = emb_data.embedding
            node_embeddings.append({**meta, "embedding": embedding})

    # Convert embeddings to a DataFrame and save to CSV
    node_embedding_df = pd.DataFrame(node_embeddings)
    #node_embedding_df.to_csv(nodes_file_name, index=False)
    node_embedding_df.to_parquet(nodes_file_name, index=False,compression='zstd')
    print(f"Embeddings saved to {nodes_file_name}")

    # Start generation

generate_embeddings(nodes,"/your/output/filepath/entity_embeddings.parquet","Enter your API key here.")
