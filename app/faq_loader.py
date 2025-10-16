import os
import uuid
import pandas as pd
import chromadb
from dotenv import load_dotenv

# Load .env for optional Cloud config
load_dotenv()

# We'll use ChromaDB without custom embedding function to avoid downloads
embedding_function = None


class FAQLoader:
    """
    Handles loading FAQs from an Excel file into ChromaDB and
    querying answers based on user questions.
    """
    def __init__(self, 
                 filename="AXA_QNA.csv", 
                 resource_dir="resource", 
                 collection_name="faq_chatbot", 
                 persist_dir="vectorstore"):
        """
        Initialize FAQLoader with filename, ChromaDB collection, and persistence directory.
        """
        # Build path relative to app directory
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
        
        # Initialize Chroma client: prefer CloudClient if env vars are set; otherwise local persistent
        chroma_api_key = os.getenv("CHROMA_API_KEY")
        chroma_tenant = os.getenv("CHROMA_TENANT")
        chroma_database = os.getenv("CHROMA_DATABASE")

        if chroma_api_key and chroma_tenant and chroma_database:
            # Cloud setup (embedding managed by server or collection settings)
            self.chroma_client = chromadb.CloudClient(
                api_key=chroma_api_key,
                tenant=chroma_tenant,
                database=chroma_database
            )
            self.collection = self.chroma_client.get_or_create_collection(
                name=collection_name
            )
        else:
            # Local persistent setup - let ChromaDB handle embeddings automatically
            os.makedirs(persist_dir, exist_ok=True)
            self.chroma_client = chromadb.PersistentClient(persist_dir)
            self.collection = self.chroma_client.get_or_create_collection(
                name=collection_name
            )

    def load_faq(self):
        """
        Load FAQs into memory for simple text matching (avoiding ChromaDB downloads).
        """
        self.faq_data = []
        for _, row in self.data.iterrows():
            question, answer = str(row.iloc[0]), str(row.iloc[1])
            self.faq_data.append({"question": question, "answer": answer})

    def query_faq(self, questions, top_k=2):
        """
        Simple text-based FAQ search using keyword matching.
        :param questions: List of user questions.
        :param top_k: Number of top results to return.
        :return: List of metadata dictionaries (answers).
        """
        if not hasattr(self, 'faq_data'):
            self.load_faq()
        
        query = questions[0].lower() if isinstance(questions, list) else questions.lower()
        results = []
        
        for faq in self.faq_data:
            question = faq["question"].lower()
            answer = faq["answer"]
            
            # Simple keyword matching
            query_words = set(query.split())
            question_words = set(question.split())
            
            # Calculate simple similarity score
            common_words = query_words.intersection(question_words)
            similarity = len(common_words) / max(len(query_words), 1)
            
            if similarity > 0.1:  # Threshold for relevance
                results.append({"answer": answer, "similarity": similarity})
        
        # Sort by similarity and return top_k
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return [{"answer": r["answer"]} for r in results[:top_k]]
