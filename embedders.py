import os
import json
import ollama
import chromadb

def query_embeddings(query_text, top_k=5):
    # Initialize ChromaDB client with persistent storage
    client = chromadb.PersistentClient(path="./chroma_db")
    
    try:
        # Try to get existing collection
        collection = client.get_collection("drug_embeddings")
    except:
        # If collection doesn't exist, create it
        collection = client.create_collection("drug_embeddings")
        
        # Regenerate embeddings if needed
        JSON_FOLDER_PATH = r"C:\Users\bharg\OneDrive\Desktop\hackathon\microlabs"
        json_files = [f for f in os.listdir(JSON_FOLDER_PATH) if f.endswith('.json')]

        for json_file in json_files:
            file_path = os.path.join(JSON_FOLDER_PATH, json_file)
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
            
            # Process and add embeddings
            if isinstance(data, dict):
                process_dict_data(data, collection, json_file)
            elif isinstance(data, list):
                process_list_data(data, collection, json_file)

    # Generate embedding for query
    query_embedding = ollama.embeddings(model='llama3:latest', prompt=query_text)['embedding']

    # Perform similarity search
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    # Extract and print results
    for i, (metadata, distance) in enumerate(zip(results['metadatas'][0], results['distances'][0]), 1):
        print(f"Result {i}:")
        print(f"  Filename: {metadata.get('filename', 'N/A')}")
        print(f"  Key/Index: {metadata.get('key', metadata.get('index', 'N/A'))}")
        print(f"  Similarity Score: {1 - distance}")  # Convert distance to similarity

def process_dict_data(data, collection, filename):
    for key, value in data.items():
        if isinstance(value, str) and value.strip():
            embedding = generate_embedding(value)
            if embedding:
                unique_id = f"{filename}_{key}"
                collection.add(
                    embeddings=[embedding],
                    metadatas=[{"filename": filename, "key": key}],
                    ids=[unique_id]
                )

def process_list_data(data, collection, filename):
    for index, item in enumerate(data):
        if isinstance(item, str) and item.strip():
            embedding = generate_embedding(item)
            if embedding:
                unique_id = f"{filename}_item_{index}"
                collection.add(
                    embeddings=[embedding],
                    metadatas=[{"filename": filename, "index": index}],
                    ids=[unique_id]
                )

def generate_embedding(text):
    try:
        response = ollama.embeddings(model='llama3:latest', prompt=text)
        return response['embedding']
    except Exception as e:
        print(f"Embedding generation error: {e}")
        return None

def main():
    query = input("Enter your search query: ")
    query_embeddings(query)

if __name__ == "__main__":
    main()