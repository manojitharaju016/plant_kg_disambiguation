import os

from datetime import datetime

import pandas as pd
import numpy as np

from openai import OpenAI



with open("path/to/api_key.txt", "r") as f:
    api_key = f.read()

def get_embedding_from_openai(text, model="text-embedding-ada-002"):
    client = OpenAI(api_key=api_key)
    response = client.embeddings.create(
        input=text,
        model=model
    )
    return response.data[0].embedding

def get_o3_mini_completion_from_openai(system_prompt, user_prompt, api_key):
    """
    Get a completion from OpenAI's O3-mini model using system and user prompts.
    
    Args:
        system_prompt (str): The system prompt to set context
        user_prompt (str): The user's input prompt
        api_key (str): OpenAI API key
        
    Returns:
        str: The model's response content
    """
    client = OpenAI(api_key=api_key)
    
    response = client.chat.completions.create(
        model="o3-mini",
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        reasoning_effort="medium"  # Options: "low", "medium", "high"
    )
    
    return response.choices[0].message.content


def get_top_10_similar_nodes(node_embedding, embedding_map_df):
    # Convert the list of embeddings to a 2D numpy array
    embeddings_array = np.array(embedding_map_df['embedding'].tolist())
    
    # Calculate cosine similarity between the node embedding and all embeddings in the map
    similarities = np.dot(embeddings_array, node_embedding) / (
        np.linalg.norm(embeddings_array, axis=1) * np.linalg.norm(node_embedding)
    )
    
    # Get the indices of the top 10 most similar nodes
    top_10_indices = np.argsort(similarities)[-10:][::-1]
    
    # Get the top 10 most similar nodes
    top_10_nodes = embedding_map_df.iloc[top_10_indices]['id'].tolist()
    
    return top_10_nodes





#def make_user_promts(row):
#    return f'Input type: {row["id"]};  Top 10 similar types: {row["top_10_nodes"]}'
print("loading embedding_map_df")
embedding_map_df = pd.read_parquet("path/to/parquet") # type_embeddings_20250311_151131.parquet
embedding_map_df.columns = ["id", "embedding"]
#define reference set as the top 30 nodes by count
reference_set = embedding_map_df["id"].head(30).tolist() # this gets the top 30 most common entity types
reference_df = embedding_map_df[embedding_map_df["id"].isin(reference_set)]

nodes_to_query = embedding_map_df[~embedding_map_df["id"].isin(reference_set)]

def get_top_10_similar_nodes(node_embedding):
    # Convert the list of embeddings to a 2D numpy array
    embeddings_array = np.array(reference_df['embedding'].tolist())
    
    # Calculate cosine similarity between the node embedding and all embeddings in the map
    similarities = np.dot(embeddings_array, node_embedding) / (
        np.linalg.norm(embeddings_array, axis=1) * np.linalg.norm(node_embedding)
    )
    
    # Get the indices of the top 10 most similar nodes
    top_10_indices = np.argsort(similarities)[-20:][::-1]
    
    # Get the top 10 most similar nodes
    top_10_nodes = reference_df.iloc[top_10_indices]['id'].tolist()
    
    return top_10_nodes


system_prompt = """
Role:
You are a senior plant biologist with deep expertise in machine learning and bioinformatics. Your task is to streamline a text-mined knowledge graph that currently contains an overly granular set of node types.

Context & Examples:
The graph currently has multiple node types for what should be broader categories. For example:

    ATR1 [gene] # special case, do not map to this unless the type is a misspelling or case difference
    ATR1 [protein] # special case, do not map to this unless the type is a misspelling or case difference
    Golgi apparatus [organelle]
    Cellulose synthesis [process]
    Pentose phosphate pathway [pathway]

Objective:
For each new node type, you will be provided with its 10 closest pre-assigned node types (determined via cosine similarity). Your goal is to assign the new node type to the most appropriate broader (parent) node type among these candidates. If the new node type does not semantically fit as a subcategory of any of the 10 candidates, then it should be designated as a new parent node.
You are also provided with the index of the new node type in the nodes_to_query dataframe.
The index goes from 1 to 3200
The lower the index, the more common the node type is, and therefore the more important it is preserved. This is important to keep in mind when making your decision.
This means that if the index is low (lower than 100 for example), you should only remap it if there is extreme confidence that it belongs to the parent node type.
For higher numbers, you can be more flexible, and here it is more important to assign it to a parent node type.
Only map to gene, protein, or enzyme if it is a misspelling or case difference.

Output Format:
Return your result as a JSON object with a single key "map", where the value is the chosen parent node type. For example:

{"map": "parent_node"}

If you determine that the new node type does not match any of the candidate parent types, return the new node type itself as the mapping value.

Instructions Recap:

    Input: A new node type and 10 closest existing node types based on cosine similarity.
    Task:
        Evaluate the semantic meaning of the new node type relative to each candidate.
        Assign it to the best-fitting parent node type if applicable.
        If the index is lower than 100, you should be extremely conservative in assigning it to a parent node type, and prefer to preserve the original node type by returning the new node type itself as the mapping value.
        When the index is higher than 100 you can be more flexible and focus on disambiguating the node type. Here it is more important to assign it to a parent node type from the list of candidates, and only if you must, return the new node type itself as the mapping value. THIS IS THE MOST IMPORTANT TASK
        If no candidate suitably represents the new node as a subcategory, mark it as a new parent node.
    Output: A JSON object in the specified format.
"""


# Initialize or load existing mapping DataFrame
mapping_file = "/home/mads/connectome/data/embeddings/type_embeddings/results/label_remap_raw.csv"
if os.path.exists(mapping_file):
    print("loading existing mapping_df")
    mapping_df = pd.read_csv(mapping_file)
    reference_df = embedding_map_df[embedding_map_df["id"].isin(mapping_df["parent_node"].unique())]
else:
    print("creating new mapping_df")
    mapping_df = pd.DataFrame(columns=["id", "parent_node"])
    # add ids from reference_df to mapping_df
    mapping_df["id"] = reference_df["id"]
    mapping_df["parent_node"] = reference_df["id"]

# Reset index of query_df
query_df = nodes_to_query.reset_index(drop=True)

from tqdm import tqdm

for index, row in tqdm(query_df.iterrows()):
    # Check if this id has already been processed
    if row["id"] in mapping_df["id"].values:
        print(f"Skipping {row['id']} - already processed")
        continue
        
    print(f"querying {row['id']}")
    print(f"length of reference_df: {len(reference_df)}")
    query_embedding = row["embedding"]
    query_frequency_rank = index
    top_10_closest_node_types = get_top_10_similar_nodes(query_embedding)
    
    input_prompt = f'Input type: {row["id"]}; Query index: {query_frequency_rank};  Top 10 similar types: {top_10_closest_node_types}'
    response = get_o3_mini_completion_from_openai(system_prompt, input_prompt, api_key)
    response = eval(response)["map"]
    
    # Add new mapping to DataFrame
    new_mapping = pd.DataFrame({"id": [row["id"]], "parent_node": [response]})
    mapping_df = pd.concat([mapping_df, new_mapping], ignore_index=True)
    
    if row["id"] == response:
        reference_df = pd.concat([
            reference_df, 
            pd.DataFrame({"id": row["id"], "embedding": [query_embedding]})
        ], ignore_index=True)
        reference_df.reset_index(drop=True, inplace=True)
        reference_df.to_csv("path/to/reference_df.csv", index=False)
    
    # Save after each iteration to preserve progress
    mapping_df.to_csv(mapping_file, index=False)



### remap labels to nodes:

def replace_label(label1, label2, label_remap_df = mapping_df):
    # replace label1 with label2 only in parent_node column in label_remap_df and return the new label_remap_df
    label_remap_df["parent_node"] = label_remap_df["parent_node"].replace(label1, label2)
    return label_remap_df

mapping_df = replace_label("webservice", "tool")
mapping_df = replace_label("difference", "other")
mapping_df = replace_label("response regulator 6", "regulation")
mapping_df = replace_label("ubiquitin-conjugating protein", "protein")
mapping_df = replace_label("cysteine protease substrate", "metabolite")
mapping_df = replace_label("agrochemicale", "other")
mapping_df = replace_label("cystatin", "protein")
mapping_df = replace_label("microscope", "tool")
mapping_df = replace_label("p12", "protein")
mapping_df = replace_label("grn", "network")
mapping_df = replace_label("tf", "transcription factor")
mapping_df = replace_label("clade", "genetic lineage")
mapping_df = replace_label("cystatin", "protein")
mapping_df = replace_label("sphingolipid", "molecule")
mapping_df = replace_label("sig", "description")
mapping_df = replace_label("viral virulence factor", "molecular effect")
mapping_df = replace_label("similar to gottwald et al., 2000", "description")
mapping_df = replace_label("protein, organism", "protein")
mapping_df = replace_label("complex", "protein complex")
mapping_df = replace_label("protein clas", "protein class")
mapping_df = replace_label("gene clas", "gene class")
mapping_df = replace_label("molecular mas", "molecular mass")
mapping_df = replace_label("consensu site", "consensus site")
mapping_df = replace_label("bioinformatic suite", "tool")
mapping_df = replace_label("cros", "cross")
mapping_df = replace_label("nucleotide binding-leucine-rich repeat protein", "protein")
mapping_df = replace_label("gene and process", "gene")
mapping_df = replace_label("calcium-dependent protein kinase", "enzyme")
mapping_df = replace_label("wheat protein", "protein")
mapping_df = replace_label("pseudoprotease", "enzyme")
mapping_df = replace_label("protein synthesi", "process")
mapping_df = replace_label("clas ius", "other")
mapping_df = replace_label("variable pair", "relationship")
mapping_df = replace_label("chemical shift", "process")


nodes_df = pd.read_csv("path/to/nodes.csv")
# in nodes_df map the :LABEL column with the label_remap_df parent_node column

nodes_df["remapped_label"] = nodes_df[":LABEL"].map(mapping_df.set_index("id")["parent_node"])

# Fix: Create a boolean mask for labels that appear less than 100 times
mask = nodes_df.groupby("remapped_label")["remapped_label"].transform("count") < 100
nodes_df.loc[mask, "remapped_label"].unique()

nodes_df.loc[mask, ":LABEL"].unique().shape

mapping_df.loc[mapping_df["id"].isin(nodes_df.loc[mask, ":LABEL"].unique()), "parent_node"] = "other"
mapping_df.to_csv("path/to/label_remap_curated.csv", index=False)













