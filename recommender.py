import streamlit as st
import ollama
import chromadb
from typing import List, Dict

class MedicationRecommender:
    def __init__(self):
        self.collection = chromadb.PersistentClient(path="./chroma_db").get_collection("drug_embeddings")
    
    def retrieve_context(self, query: str, top_k: int = 3) -> List[Dict]:
        # Embedding generation
        query_embedding = ollama.embeddings(model='llama3:latest', prompt=query)['embedding']
        
        # Semantic search
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        contexts = []
        for metadata in results['metadatas'][0]:
            contexts.append({
                'filename': metadata.get('filename', 'Unknown'),
                'key': metadata.get('key', metadata.get('index', 'Unknown'))
            })
        
        return contexts

    def rag_recommendation(self, query: str) -> str:
        contexts = self.retrieve_context(query)
        
        augmented_prompt = f"""
        Context Information:
        {contexts}

        Medical Query: {query}

        As a medical assistant, provide:
        1. Safety assessment of the medication
        2. Potential risks or interactions
        3. Alternative medication recommendations
        4. Reasoning for recommendations
        """
        
        response = ollama.chat(
            model='llama3:latest',
            messages=[
                {"role": "system", "content": "You are a precise medical recommendation assistant."},
                {"role": "user", "content": augmented_prompt}
            ]
        )
        
        return response['message']['content']

    def agent_recommendation(self, query: str) -> str:
        # Agent-based approach with multiple reasoning steps
        steps = [
            f"Analyze safety of medication for: {query}",
            "Check potential interactions",
            "Evaluate patient-specific risks",
            "Generate alternative recommendations"
        ]
        
        comprehensive_response = []
        for step in steps:
            agent_response = ollama.chat(
                model='llama3:latest',
                messages=[
                    {"role": "system", "content": "You are a medical reasoning agent."},
                    {"role": "user", "content": step}
                ]
            )
            comprehensive_response.append(agent_response['message']['content'])
        
        return "\n\n".join(comprehensive_response)

def streamlit_app():
    st.title("Medication Recommender System")
    
    recommender = MedicationRecommender()
    
    query = st.text_input("Enter medication/health query:")
    method = st.radio("Recommendation Method", ["RAG", "Agent-based"])
    
    if query:
        with st.spinner('Generating recommendation...'):
            if method == "RAG":
                recommendation = recommender.rag_recommendation(query)
            else:
                recommendation = recommender.agent_recommendation(query)
            
            st.subheader("Recommendation:")
            st.write(recommendation)

if __name__ == "__main__":
    streamlit_app()