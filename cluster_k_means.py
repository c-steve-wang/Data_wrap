import json
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

def cluster_instructions(input_path, output_path):
    # Load the data from the input JSON file
    with open(input_path, 'r') as f:
        data = json.load(f)

    # Extract instructions from the data
    instructions = [item['instruction'] for item in data]

    # Initialize the sentence transformer model
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    # Generate embeddings for each instruction
    embeddings = model.encode(instructions)

    # Use K-means to cluster the embeddings into 50 clusters
    kmeans = KMeans(n_clusters=50, random_state=0).fit(embeddings)

    # Assign cluster IDs to each item based on the K-means clustering
    for i, item in enumerate(data):
        item['cluster_id'] = int(kmeans.labels_[i])

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=4)

input_path = 'concatenated_data.json'
output_path = 'clustered_data.json'

cluster_instructions(input_path, output_path)
