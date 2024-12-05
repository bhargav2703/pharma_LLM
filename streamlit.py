import streamlit as st
import ollama
import chromadb

def initialize_chromadb():
    client = chromadb.PersistentClient(path="./chroma_db")
    return client.get_collection("drug_embeddings")

def generate_embedding(query):
    return ollama.embeddings(model='llama3:latest', prompt=query)['embedding']

def retrieve_context(collection, query_embedding, top_k=3):
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    contexts = []
    for metadata in results['metadatas'][0]:
        filename = metadata.get('filename', 'Unknown')
        key = metadata.get('key', metadata.get('index', 'Unknown'))
        contexts.append(f"Source: {filename}, Key: {key}")
    return contexts

def rag_query(collection, query_text):
    query_embedding = generate_embedding(query_text)
    contexts = retrieve_context(collection, query_embedding)

    augmented_prompt = f"""
    Context: 
    {' '.join(contexts)}

    Question: {query_text}

    Using the above context, provide a detailed and informative answer.
    """

    response = ollama.chat(
        model='llama3:latest',
        messages=[
            {"role": "system", "content": "You are a helpful assistant that uses provided context to answer questions precisely."},
            {"role": "user", "content": augmented_prompt}
        ]
    )

    return response['message']['content'], contexts

def main():
    st.title("Drug Information RAG Assistant")
    
    # Initialize ChromaDB collection
    collection = initialize_chromadb()

    # User input
    query = st.text_input("Enter your medical/drug-related query:")
    
    if query:
        # Generate response
        with st.spinner('Generating response...'):
            answer, contexts = rag_query(collection, query)
        
        # Display results
        st.subheader("Answer:")
        st.write(answer)
        
        # Show context sources
        st.subheader("Context Sources:")
        for context in contexts:
            st.markdown(f"- {context}")

if __name__ == "__main__":
    main()