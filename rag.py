import ollama
import chromadb

def rag_query(query_text, top_k=3):
    # Initialize ChromaDB client
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_collection("drug_embeddings")

    # Generate embedding for query
    query_embedding = ollama.embeddings(model='llama3:latest', prompt=query_text)['embedding']

    # Retrieve relevant context
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    # Extract context from retrieved documents
    contexts = []
    for metadata in results['metadatas'][0]:
        filename = metadata.get('filename', 'Unknown')
        key = metadata.get('key', metadata.get('index', 'Unknown'))
        contexts.append(f"Source: {filename}, Key: {key}")

    # Construct prompt with retrieved context
    augmented_prompt = f"""
    Context: 
    {' '.join(contexts)}

    Question: {query_text}

    Using the above context, provide a detailed and informative answer.
    """

    # Generate response using Ollama
    response = ollama.chat(
        model='llama3:latest',
        messages=[
            {"role": "system", "content": "You are a helpful assistant that uses provided context to answer questions precisely."},
            {"role": "user", "content": augmented_prompt}
        ]
    )

    return response['message']['content']

def main():
    while True:
        query = input("Enter your query (or 'quit' to exit): ")
        if query.lower() == 'quit':
            break
        
        answer = rag_query(query)
        print("\nAnswer:")
        print(answer)
        print("\n" + "-"*50 + "\n")

if __name__ == "__main__":
    main()