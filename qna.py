import ollama
import chromadb

def query_embeddings(query_text, top_k=5):
    # Initialize ChromaDB client
    client = chromadb.Client()
    collection = client.get_collection("drug_embeddings")

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

def main():
    query = input("Enter your search query: ")
    query_embeddings(query)

if __name__ == "__main__":
    main()