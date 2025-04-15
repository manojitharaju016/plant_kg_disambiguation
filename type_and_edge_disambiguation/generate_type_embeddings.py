
import os
import json
import time
from datetime import datetime

import pandas as pd
import numpy as np
import tiktoken
import gzip
from openai import OpenAI


def prepare_embedding_data(array_of_strings):
    """Prepare GO terms data for embedding."""
    encoding = tiktoken.get_encoding("cl100k_base")
    embedding_data = []
    total_tokens = 0

    for edge in array_of_strings:
        text = edge
        n_tokens = len(encoding.encode(text))
        total_tokens += n_tokens
        
        embedding_data.append({
            'id': edge,
            'text': text,
            'n_tokens': n_tokens
        })

    estimated_cost = (total_tokens / 1000000) * 0.1
    print(f"Total number of submissions: {len(embedding_data)}")
    print(f"Total tokens: {total_tokens}")
    print(f"Estimated cost: ${estimated_cost:.2f}")
    
    return embedding_data


def create_batch_file(embedding_data, query_dir, timestamp, chunk_number=None):
    """Create JSONL file for batch submission."""
    batch_fname = f'embedding_requests_{timestamp}.jsonl'
    batch_file_path = os.path.join(query_dir, batch_fname)
    if chunk_number is not None:
        batch_fname = f'embedding_requests_{timestamp}_chunk_{chunk_number}.jsonl'
        batch_file_path = os.path.join(query_dir, batch_fname)
    with open(batch_file_path, 'w') as f:
        for item in embedding_data:
            request = {
                "custom_id": item['id'],
                "method": "POST", 
                "url": "/v1/embeddings",
                "body": {
                    "model": "text-embedding-ada-002",
                    "input": item['text']
                }
            }
            f.write(json.dumps(request) + '\n')
    
    return batch_file_path


def create_multiple_batch_files(client,embeddings_array, timestamp, query_dir, max_batch_size=50_000, description="Edge embeddings batch job"):
    """Create multiple batch files for embedding."""
    
    #split embeddings_array into chunks of max_batch_size
    chunks = [embeddings_array[i:i+max_batch_size] for i in range(0, len(embeddings_array), max_batch_size)]
    
    batch_ids = []
    batch_input_files = []
    batch_file_paths = []
    for i, chunk in enumerate(chunks):
        batch_file_path = create_batch_file(chunk, query_dir, timestamp, chunk_number=i)
        batch_file_paths.append(batch_file_path)
        batch, batch_input_file = submit_batch_job(client, batch_file_path, description)
        batch_ids.append(batch.id)
        batch_input_files.append(batch_input_file)
        
    return batch_ids, batch_input_files, batch_file_paths
    

def submit_batch_job(client, batch_file_path, description):
    """Submit batch job to OpenAI API."""
    print("submitting batch job")
    batch_input_file = client.files.create(
        file=open(batch_file_path, "rb"),
        purpose="batch"
    )
    #check if file is uploaded successfully
    while batch_input_file.status != "processed":
        print(f"batch input file status: {batch_input_file.status}")
        time.sleep(1)
    
    print(f"batch input file created\n{batch_input_file.id}")
    batch = client.batches.create(
        input_file_id=batch_input_file.id,
        endpoint="/v1/embeddings",
        completion_window="24h",
        metadata={"description": description}
    )
    print(f"batch created\n{batch.id}")
    return batch, batch_input_file

def wait_for_completion(client, batch_id, initial_pause=4):
    """Wait for batch job completion with exponential backoff."""
    pause = initial_pause
    completed = False
    while not completed:
        batch = client.batches.retrieve(batch_id)
        print(batch.status)
        
        if batch.status == "completed":
            print("Batch completed")
            completed = True
            return batch
        
        print(f"Batch not completed, pausing for {pause} seconds")
        time.sleep(pause)
        pause *= 2
        
def wait_for_completion_for_multiple_batches(client, batch_ids, initial_pause=4):
    """Wait for completion of multiple batch jobs with exponential backoff."""
    pause = initial_pause
    all_completed = False
    #mark all batches as not completed
    batch_statuses_dict = {batch_id: "bla" for batch_id in batch_ids}
    batches = []
    while all_completed == False:
        for batch_id in batch_ids:
            if batch_statuses_dict[batch_id] != "completed":
                batch = client.batches.retrieve(batch_id)
                if batch.status == "completed":
                    batch_statuses_dict[batch_id] = batch.status
                    batches.append(batch)
                else:
                    batch_statuses_dict[batch_id] = batch.status
                
        #check if all batches are completed
        if all(status == "completed" for status in batch_statuses_dict.values()):
            print("All batches completed")
            print(batch_statuses_dict)
            all_completed = True
            return batches
        
        
        print(batch_statuses_dict)
        
        if pause > 300:
            pause = 300
            
        print(f"Batch not completed, pausing for {pause} seconds")
        time.sleep(pause)
        pause *= 2

def process_output(client, batch, output_dir, timestamp, batch_file_path, chunk_number=None):
    """Process and save batch job output."""

    # Save output
    output_fname = f'embedding_output_{timestamp}'
    if chunk_number is not None:
        output_fname = f'embedding_output_{timestamp}_chunk_{chunk_number}.jsonl'
    output_path = os.path.join(output_dir, output_fname)
    if os.path.exists(output_path):
        print(f"output file already exists, skipping {output_path}")
        return output_path


    batch_output_file = client.files.content(batch.output_file_id)
    content = batch_output_file.text
    split_content = content.split("\n")[:-1]
    
    print(f"batch output file saved to {output_path}")
    with open(output_path, 'w') as f:
        for line in split_content:
            f.write(line + '\n')

    # Save batch info
    batch_info = {
        "batch_file_path": batch_file_path,
        "batch_id": batch.id,
        "output_file_id": batch.output_file_id
    }
    
    log_fname = f'embedding_log_{timestamp}'
    if chunk_number is not None:
        log_fname = f'embedding_log_{timestamp}_chunk_{chunk_number}.json'
    log_path = os.path.join(output_dir, log_fname)
    
    with open(log_path, 'w') as f:
        json.dump(batch_info, f, indent=4)

    print(f"batch info saved to {log_path}")
    return output_path



def process_output_for_multiple_batches(client, batch_ids, batch_file_paths, output_dir, timestamp):
    """Process output for multiple batches."""
    i = 0
    output_paths = []
    for batch_id, batch_file_path in zip(batch_ids, batch_file_paths):
        batch = client.batches.retrieve(batch_id)
        output_path = process_output(client, batch, output_dir, timestamp, batch_file_path, chunk_number=i)

        i += 1
        output_paths.append(output_path)
    
    return output_paths
    
    



def get_embedding_from_openai(text, model="text-embedding-ada-002", api_key=None):
    client = OpenAI(api_key=api_key)
    response = client.embeddings.create(
        input=text,
        model=model
    )
    return response.data[0].embedding

def load_output_jsonl(file_path, client=None, api_key=None):
    # Read the gzipped JSONL file and parse each line as JSON
    print(file_path)
    with open(file_path, 'rt') as f:  # Changed to gzip.open with text mode
        data = []
        for line in f:
            try:
                json_obj = json.loads(line)
                embedding = json_obj['response']['body']['data'][0]['embedding']
                if len(embedding) != 1536:
                    print(f"Embedding length is not 1536 for line: {line}")
                    embedding = get_embedding_from_openai(json_obj['custom_id'], api_key=api_key)
                data.append({
                    'id': json_obj['custom_id'],
                    'embedding': embedding
                })
            except Exception as e:
                print(f"Error processing line: {e}")
                # If client is provided, try to get embedding directly
                if client:
                    try:
                        json_obj = json.loads(line)
                        input_text = json_obj['custom_id']
                        
                        embedding = get_embedding_from_openai(input_text, api_key=api_key)
                        #check that the embedding is 1536
                        if len(embedding) != 1536:
                            print(f"Embedding length is not 1536 for line: {line}")
                            continue
                        data.append({
                            'id': input_text,
                            'embedding': embedding
                        })
                        print(f"Successfully retrieved embedding for {input_text}")
                    except Exception as e2:
                        print(f"Failed to get embedding directly: {e2}")
                        print("failed again for:", line)
                        continue
                else:
                    print("No client provided to retry failed embedding")
                    continue
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    return df

def load_output_for_multiple_batches(output_paths, client=None, api_key=None):
    """Load output for multiple batches."""
    output_dfs = pd.concat([load_output_jsonl(output_path, client, api_key) for output_path in output_paths])
    return output_dfs



# Load API key for OpenAI
with open("data/api_key.txt", "r") as f:
    api_key = f.read()
# Initialize OpenAI client
client = OpenAI(api_key=api_key)

#get input from user wether they want to run from scratch or load existing batch
run_from_scratch = input("Do you want to run from scratch? (y/n): ")
assert run_from_scratch in ["y", "n"], "Invalid input, please enter y or n"
if run_from_scratch == "y":

    nodes_df = pd.read_json('/home/mads/connectome/data/embeddings/type_embeddings/results/entityedgebasisall_dic.json')

    
    types1 = nodes_df["entity1type"].astype(str).unique()
    types2 = nodes_df["entity2type"].astype(str).unique()

    #combine types1 and types2
    types = np.concatenate([types1, types2])

    #remove duplicates
    types = np.unique(types)

    



    inputs = list(types)

    print(f"Calculating embeddings for {len(inputs)} unique connection types")


    # Prepare embedding data from unique connection types
    embeddings_array = prepare_embedding_data(inputs)


    # Generate a timestamp for file identification
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Timestamp file identifier: {timestamp}")

    # Set parameters for batch processing
    batch_size = 45000
    query_dir = "data/embeddings/type_embeddings/queries"

    # Check if the query directory exists; if not, create it
    if not os.path.exists(query_dir):
        os.makedirs(query_dir)

    # Create multiple batch files for processing
    batch_ids, batch_input_files, batch_file_paths = create_multiple_batch_files(
        client,
        embeddings_array,
        timestamp,
        query_dir,
        max_batch_size=batch_size,
        description="Edge embeddings batch job"
    )

    # Save batch IDs and file paths to a DataFrame and export to CSV
    batch_df = pd.DataFrame({"batch_id": batch_ids, "batch_file_path": batch_file_paths, "batch_input_file": batch_input_files})
    batch_df.to_csv(f"data/embeddings/type_embeddings/batch_ids_{timestamp}.csv", index=False)



#check if variable timestamp exists
if run_from_scratch == "n":
    #get input from user
    timestamp = input("Enter the timestamp that you used for the batch job: ")

    batch_df = pd.read_csv(f"data/embeddings/type_embeddings/batch_ids_{timestamp}.csv")
    batch_ids = batch_df["batch_id"].tolist()
    batch_file_paths = batch_df["batch_file_path"].tolist()

print("Waiting for batch jobs to complete")
batches = wait_for_completion_for_multiple_batches(client, batch_ids)
print("Batch jobs completed")


# Set output directory for results
output_dir = "data/embeddings/type_embeddings/output"

# Check if the output directory exists; if not, create it
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Process output for multiple batches and save results
print(f"Processing output for multiple batches, saving to jsonl files in {output_dir}")
output_paths = process_output_for_multiple_batches(
    client,
    batch_ids,
    batch_file_paths,
    output_dir,
    timestamp
)

print(f"Loading output for multiple batches, saving to parquet in {output_dir}")
output_dfs = load_output_for_multiple_batches(output_paths, client, api_key)
#reset index
output_dfs.reset_index(drop=True, inplace=True)

def get_next_available_filename(base_path):
    """
    Returns the next available filename by adding _n suffix if file exists.
    Example: if file.parquet exists, returns file_1.parquet
    """
    if not os.path.exists(base_path):
        return base_path
        
    directory = os.path.dirname(base_path)
    filename = os.path.basename(base_path)
    name, ext = os.path.splitext(filename)
    
    counter = 1
    while os.path.exists(base_path):
        new_filename = f"{name}_{counter}{ext}"
        base_path = os.path.join(directory, new_filename)
        counter += 1
        
    return base_path

# Original parquet save code modified to use the new function
parquet_output_path = os.path.join(output_dir, f"type_embeddings_{timestamp}.parquet")
final_output_path = get_next_available_filename(parquet_output_path)
output_dfs.to_parquet(final_output_path)
