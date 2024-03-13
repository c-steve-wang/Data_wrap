import json
import random
from transformers import AutoTokenizer

def create_cluster_chunks(input_path, output_path, max_length=2048):
    # Load the clustered data
    with open(input_path, 'r') as file:
        data = json.load(file)

    # Group data by cluster_id
    clusters = {}
    for item in data:
        cluster_id = item['cluster_id']
        if cluster_id not in clusters:
            clusters[cluster_id] = []
        clusters[cluster_id].append(item)

    all_chunks = [] 

    # Initialize the tokenizer
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')

    for cluster_id, items in clusters.items():
        i = 0
        while i < len(items):
            chunk_size = random.randint(1, 10)  # Determine random chunk size
            chunk = []  # To store selected items for the current chunk

            while len(chunk) < chunk_size and i < len(items):
                item = items[i]
                # Prepare potential new instructions and outputs
                new_instructions = "\n\n".join([x['instruction'] for x in chunk + [item]])
                new_outputs = "\n\n".join([(x['output'] if x['output'].strip() else " ") for x in chunk + [item]])

                # Check if adding the new item exceeds the max_length
                if len(tokenizer.tokenize(new_instructions)) + len(tokenizer.tokenize(new_outputs)) <= max_length:
                    chunk.append(item)  # Add item to the chunk
                    i += 1  # Move to the next item
                else:
                    # If the chunk is empty and a single item exceeds the limit, add it anyway to avoid infinite loop
                    if not chunk:
                        chunk.append(item)
                        i += 1  # Move to the next item
                    break  # Proceed to create the chunk

            # Concatenate instructions and outputs for the current chunk
            instructions_inputs = "\n\n".join([x['instruction'] for x in chunk]).strip()
            outputs = "\n\n".join([x['output'] if x['output'].strip() else " " for x in chunk]).strip()

            concatenated_group = {
                "instruction": instructions_inputs,
                "output": outputs
            }
            all_chunks.append(concatenated_group)

    # Save all chunks to the output JSON file
    with open(output_path, 'w') as outfile:
        json.dump(all_chunks, outfile, indent=4)

# Specify paths and call the function
input_path = 'cluster_new.json'  # Adjust as needed
output_path = 'clustered_data.json'  # Adjust as needed
create_cluster_chunks(input_path, output_path)
