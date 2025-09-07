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

    def __init__(self, file_path="resource/FAQ_Nawa.xlsx", collection_name="faq_nawa", persist_dir="vectorstore"):
        """
        Initialize FAQLoader with file path, ChromaDB collection, and persistence directory.
        """
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
