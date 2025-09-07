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
                 filename="FAQ_Nawa.xlsx", 
                 resource_dir="resource", 
                 collection_name="faq_nawa", 
                 persist_dir="vectorstore"):
        """
        Initialize FAQLoader with filename, ChromaDB collection, and persistence directory.
        """
        # Build path relative to repo root
        root_dir = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.dirname(root_dir)
        file_path = os.path.join(repo_root, resource_dir, filename)

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"FAQ file not found at: {file_path}")
        
        self.file_path = file_path
        self.data = pd.read_excel(file_path)

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
                question, answer = str(row["Question"]), str(row["Answer"])
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
