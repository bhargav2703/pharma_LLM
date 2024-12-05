import streamlit as st
import ollama
import chromadb
import json
import os

class ImprovedSummarizer:
    def __init__(self, json_folder_path):
        self.json_folder_path = json_folder_path
        self.collection = chromadb.PersistentClient(path="./chroma_db").get_collection("drug_embeddings")

    def load_json_files(self):
        json_files = [f for f in os.listdir(self.json_folder_path) if f.endswith('.json')]
        all_data = []
        
        for file in json_files:
            with open(os.path.join(self.json_folder_path, file), 'r', encoding='utf-8') as f:
                data = json.load(f)
                all_data.append({
                    'filename': file,
                    'content': json.dumps(data)
                })
        
        return all_data

    def semantic_search(self, query, top_k=3):
        query_embedding = ollama.embeddings(model='llama3:latest', prompt=query)['embedding']
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        return results

    def summarize(self, query):
        # Semantic search
        search_results = self.semantic_search(query)
        
        # Prepare context
        context_prompt = "Relevant information:\n"
        for metadata in search_results['metadatas'][0]:
            filename = metadata.get('filename', 'Unknown')
            context_prompt += f"From {filename}: {metadata}\n"
        
        # Generate summary
        summary_prompt = f"""
        Query: {query}
        {context_prompt}

        Provide a precise, informative summary addressing the query. 
        Steps:
        1. Extract key information relevant to the query
        2. Synthesize a concise summary
        3. Highlight main points
        """

        response = ollama.chat(
            model='llama3:latest',
            messages=[
                {"role": "system", "content": "You are an expert summarization assistant."},
                {"role": "user", "content": summary_prompt}
            ]
        )

        return response['message']['content']

def streamlit_summarizer():
    st.title("Advanced Document Summarizer")
    
    # Adjust path to your JSON folder
    json_folder_path = r"C:\Users\bharg\OneDrive\Desktop\hackathon\microlabs"
    
    summarizer = ImprovedSummarizer(json_folder_path)
    
    query = st.text_input("Enter your summarization query:")
    
    if query:
        with st.spinner('Generating summary...'):
            summary = summarizer.summarize(query)
            
            st.subheader("Summary:")
            st.write(summary)

if __name__ == "__main__":
    streamlit_summarizer()