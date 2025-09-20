import os
import uuid
import pandas as pd
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction


# Define embedding function
embedding_function = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")


class FAQLoader:
    """
    Handles loading FAQs from an Excel file into ChromaDB and
    querying answers based on user questions.
    """
    def __init__(self, 
                 filename="medquad.csv", 
                 resource_dir="resource", 
                 collection_name="faq_chatbot", 
                 persist_dir="vectorstore"):
        """
        Initialize FAQLoader with filename, ChromaDB collection, and persistence directory.
        """
        # Build path relative to repo root
        app_root = os.path.dirname(os.path.abspath(__file__))        
        file_path = os.path.join(app_root, resource_dir, filename)

        if not os.path.exists(file_path):
            print("DEBUG: looking for file at", file_path)
            raise FileNotFoundError(f"FAQ file not found at: {file_path}")
        
        self.file_path = file_path
        # check file extension
        ext = os.path.splitext(file_path)[1].lower()  
        if ext == ".xlsx":
            self.data = pd.read_excel(file_path, nrows=100)
        elif ext == ".csv":
            self.data = pd.read_csv(file_path, nrows=100)
        else:
            raise ValueError(f"Unsupported file extension: {ext}")
        
        os.makedirs(persist_dir, exist_ok=True)
        
        self.chroma_client = chromadb.PersistentClient(persist_dir)
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            embedding_function=embedding_function
        )

    def load_faq(self):
        """
        Load FAQs into the ChromaDB collection if not already present.
        """
        if self.collection.count() == 0:
            for _, row in self.data.iterrows():
                question, answer = str(row.iloc[0]), str(row.iloc[1]) 
                self.collection.add(
                    documents=[question],
                    metadatas=[{"answer": answer}],
                    ids=[str(uuid.uuid4())]
                )

    def query_faq(self, questions, top_k=2):
        """
        Query the collection for the most relevant answers.
        :param questions: List of user questions.
        :param top_k: Number of top results to return.
        :return: List of metadata dictionaries (answers).
        """
        results = self.collection.query(query_texts=questions, n_results=top_k)
        return results.get('metadatas', [])
