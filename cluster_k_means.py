import json
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.cluster import KMeans
import numpy as np

def cluster_instructions(input_path, output_path):
    # Load the data from the input JSON file
    with open(input_path, 'r') as f:
        data = json.load(f)

    # Extract instructions from the data
    instructions = [item['instruction'] for item in data]

    # Initialize the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Using device: {device}")

    embeddings = []
    # Process each instruction
    for instruction in instructions:
        # Tokenize the instruction and move tensors to the correct device
        inputs = tokenizer(instruction, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Forward pass, get hidden states
        with torch.no_grad():
            outputs = model(**inputs)

        # Get the embeddings from the last hidden state
        # The embeddings of all tokens are averaged to get a single vector per instruction
        last_hidden_states = outputs.last_hidden_state
        embeddings.append(last_hidden_states.mean(dim=1).cpu().numpy())

    # Convert embeddings list to a 2D array
    embeddings = np.vstack(embeddings)

    # Use K-means to cluster the embeddings into 50 clusters
    kmeans = KMeans(n_clusters=50, random_state=0).fit(embeddings)

    # Assign cluster IDs to each item based on the K-means clustering
    for i, item in enumerate(data):
        item['cluster_id'] = int(kmeans.labels_[i])

    # Save the data with cluster IDs to the output JSON file
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=4)

# Specify the paths to your input and output JSON files
input_path = 'concatenated_data.json'
output_path = 'cluster_new.json'

# Perform the clustering and assign cluster IDs
cluster_instructions(input_path, output_path)

