"""
ChromaDB Management CLI
Simple command-line tool to view and manage ChromaDB collections
"""

import os
import chromadb
from dotenv import load_dotenv
from faq_loader import E5EmbeddingFunction

# Load environment variables
load_dotenv()

class ChromaDBManager:
    def __init__(self):
        """Initialize ChromaDB client"""
        # Initialize embedding function
        self.embedding_function = E5EmbeddingFunction()
        
        # Initialize Chroma client
        chroma_api_key = os.getenv("CHROMA_API_KEY")
        chroma_tenant = os.getenv("CHROMA_TENANT")
        chroma_database = os.getenv("CHROMA_DATABASE")

        if chroma_api_key and chroma_tenant and chroma_database:
            # Cloud setup
            self.chroma_client = chromadb.CloudClient(
                api_key=chroma_api_key,
                tenant=chroma_tenant,
                database=chroma_database
            )
            print("✓ Connected to ChromaDB Cloud")
        else:
            # Local persistent setup (same path as faq_loader.py)
            persist_path = os.path.join(os.path.dirname(__file__), "vectorstore")
            self.chroma_client = chromadb.PersistentClient(persist_path)
            print("✓ Connected to ChromaDB Local")
        
        # Collection names
        self.collection_names = ["axa_product_sales", "axa_customer_corporate"]
    
    def view_data(self):
        """View all data from collections"""
        print("\n" + "="*70)
        print("VIEWING ALL DATA FROM CHROMADB")
        print("="*70)
        
        total_records = 0
        
        for collection_name in self.collection_names:
            try:
                collection = self.chroma_client.get_collection(
                    name=collection_name,
                    embedding_function=self.embedding_function
                )
                
                # Get all data from collection
                results = collection.get()
                count = len(results['ids'])
                total_records += count
                
                print(f"\n{'─'*70}")
                print(f"COLLECTION: {collection_name}")
                print(f"{'─'*70}")
                print(f"Total records: {count}")
                
                if count > 0:
                    print("\nRecords:")
                    for i, (id, document, metadata) in enumerate(zip(
                        results['ids'],
                        results['documents'],
                        results['metadatas']
                    ), 1):
                        print(f"\n[{i}] ID: {id}")
                        print(f"    Document: {document[:150]}..." if len(document) > 150 else f"    Document: {document}")
                        
                        # Display structured metadata
                        if metadata.get('question_original'):
                            print(f"    Question: {metadata.get('question_original')}")
                        if metadata.get('category_topic'):
                            print(f"    Category: {metadata.get('category_topic')}")
                        if metadata.get('product_name'):
                            print(f"    Product: {metadata.get('product_name')}")
                        if metadata.get('insurance_type'):
                            print(f"    Insurance Type: {metadata.get('insurance_type')}")
                        if metadata.get('topic_focus'):
                            print(f"    Topic: {metadata.get('topic_focus')}")
                        if metadata.get('coverage_keyword'):
                            print(f"    Coverage: {metadata.get('coverage_keyword')}")
                        if metadata.get('action_type'):
                            print(f"    Action: {metadata.get('action_type')}")
                        if metadata.get('entity'):
                            print(f"    Entity: {metadata.get('entity')}")
                        if metadata.get('doc_type'):
                            print(f"    Doc Type: {metadata.get('doc_type')}")
                        if metadata.get('source'):
                            print(f"    Source: {metadata.get('source')}")
                else:
                    print("(Collection is empty)")
                    
            except Exception as e:
                print(f"\n✗ Error accessing collection '{collection_name}': {e}")
        
        print("\n" + "="*70)
        print(f"TOTAL RECORDS ACROSS ALL COLLECTIONS: {total_records}")
        print("="*70 + "\n")
    
    def delete_all_data(self):
        """Delete all data from collections"""
        print("\n" + "="*70)
        print("⚠️  WARNING: DELETE ALL DATA")
        print("="*70)
        
        confirm = input("\nAre you sure you want to delete ALL data? (yes/no): ").strip().lower()
        
        if confirm != 'yes':
            print("✓ Deletion cancelled")
            return
        
        # Double confirmation
        confirm2 = input("Type 'DELETE' to confirm: ").strip()
        
        if confirm2 != 'DELETE':
            print("✓ Deletion cancelled")
            return
        
        print("\nDeleting all data...")
        
        for collection_name in self.collection_names:
            try:
                # Delete the collection
                self.chroma_client.delete_collection(name=collection_name)
                print(f"✓ Deleted collection: {collection_name}")
                
                # Recreate empty collection
                self.chroma_client.create_collection(
                    name=collection_name,
                    embedding_function=self.embedding_function
                )
                print(f"✓ Recreated empty collection: {collection_name}")
                
            except Exception as e:
                print(f"✗ Error deleting collection '{collection_name}': {e}")
        
        print("\n✓ All data deleted successfully!\n")
    
    def search_by_metadata(self):
        """Search/filter data by metadata fields"""
        print("\n" + "="*70)
        print("SEARCH BY METADATA")
        print("="*70)
        
        print("\nAvailable filters:")
        print("1. Product Name")
        print("2. Insurance Type")
        print("3. Category Topic")
        print("4. Action Type")
        print("5. Entity")
        print("6. Back to main menu")
        
        choice = input("\nSelect filter type (1-6): ").strip()
        
        if choice == '6':
            return
        
        filter_map = {
            '1': ('product_name', 'Product Name'),
            '2': ('insurance_type', 'Insurance Type'),
            '3': ('category_topic', 'Category Topic'),
            '4': ('action_type', 'Action Type'),
            '5': ('entity', 'Entity')
        }
        
        if choice not in filter_map:
            print("\n✗ Invalid option")
            return
        
        field_name, display_name = filter_map[choice]
        filter_value = input(f"\nEnter {display_name} to search for: ").strip()
        
        if not filter_value:
            print("\n✗ Search value cannot be empty")
            return
        
        print(f"\n{'='*70}")
        print(f"SEARCHING: {display_name} = '{filter_value}'")
        print(f"{'='*70}")
        
        total_found = 0
        
        for collection_name in self.collection_names:
            try:
                collection = self.chroma_client.get_collection(
                    name=collection_name,
                    embedding_function=self.embedding_function
                )
                
                # Query with where filter
                try:
                    results = collection.get(
                        where={field_name: filter_value}
                    )
                    count = len(results['ids'])
                    total_found += count
                    
                    if count > 0:
                        print(f"\n{'─'*70}")
                        print(f"COLLECTION: {collection_name} - Found {count} record(s)")
                        print(f"{'─'*70}")
                        
                        for i, (id, document, metadata) in enumerate(zip(
                            results['ids'],
                            results['documents'],
                            results['metadatas']
                        ), 1):
                            print(f"\n[{i}] ID: {id}")
                            print(f"    Document: {document[:150]}..." if len(document) > 150 else f"    Document: {document}")
                            
                            # Display structured metadata
                            if metadata.get('question_original'):
                                print(f"    Question: {metadata.get('question_original')}")
                            if metadata.get('category_topic'):
                                print(f"    Category: {metadata.get('category_topic')}")
                            if metadata.get('product_name'):
                                print(f"    Product: {metadata.get('product_name')}")
                            if metadata.get('insurance_type'):
                                print(f"    Insurance Type: {metadata.get('insurance_type')}")
                            if metadata.get('topic_focus'):
                                print(f"    Topic: {metadata.get('topic_focus')}")
                            if metadata.get('coverage_keyword'):
                                print(f"    Coverage: {metadata.get('coverage_keyword')}")
                            if metadata.get('action_type'):
                                print(f"    Action: {metadata.get('action_type')}")
                            if metadata.get('entity'):
                                print(f"    Entity: {metadata.get('entity')}")
                
                except Exception as e:
                    print(f"\n⚠️  Metadata filter not supported or error: {e}")
                    print(f"   Trying alternative method...")
                    
                    # Fallback: get all and filter manually
                    results = collection.get()
                    filtered_ids = []
                    filtered_docs = []
                    filtered_meta = []
                    
                    for id, doc, meta in zip(results['ids'], results['documents'], results['metadatas']):
                        if meta.get(field_name) == filter_value:
                            filtered_ids.append(id)
                            filtered_docs.append(doc)
                            filtered_meta.append(meta)
                    
                    count = len(filtered_ids)
                    total_found += count
                    
                    if count > 0:
                        print(f"\n{'─'*70}")
                        print(f"COLLECTION: {collection_name} - Found {count} record(s)")
                        print(f"{'─'*70}")
                        
                        for i, (id, document, metadata) in enumerate(zip(filtered_ids, filtered_docs, filtered_meta), 1):
                            print(f"\n[{i}] ID: {id}")
                            print(f"    Document: {document[:150]}..." if len(document) > 150 else f"    Document: {document}")
                            print(f"    {display_name}: {metadata.get(field_name)}")
                            
            except Exception as e:
                print(f"\n✗ Error searching collection '{collection_name}': {e}")
        
        print(f"\n{'='*70}")
        print(f"TOTAL FOUND: {total_found} record(s)")
        print(f"{'='*70}\n")
    
    def show_menu(self):
        """Display CLI menu"""
        print("\n" + "="*70)
        print("CHROMADB MANAGEMENT TOOL")
        print("="*70)
        print("\n1. View all data")
        print("2. Delete all data")
        print("3. Search by metadata")
        print("4. Exit")
        print()
    
    def run(self):
        """Main CLI loop"""
        while True:
            self.show_menu()
            
            try:
                choice = input("Select option (1-4): ").strip()
                
                if choice == '1':
                    self.view_data()
                elif choice == '2':
                    self.delete_all_data()
                elif choice == '3':
                    self.search_by_metadata()
                elif choice == '4':
                    print("\nExiting... Goodbye! 👋\n")
                    break
                else:
                    print("\n✗ Invalid option. Please select 1, 2, 3, or 4.\n")
                    
            except KeyboardInterrupt:
                print("\n\nExiting... Goodbye! 👋\n")
                break
            except Exception as e:
                print(f"\n✗ An error occurred: {e}\n")


if __name__ == "__main__":
    try:
        manager = ChromaDBManager()
        manager.run()
    except Exception as e:
        print(f"\n✗ Failed to initialize ChromaDB Manager: {e}\n")

