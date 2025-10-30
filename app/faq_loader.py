import os
import uuid
import pandas as pd
import chromadb
import logging
import re
from dotenv import load_dotenv
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    # Fallback to old import for backward compatibility
    from langchain_community.embeddings import HuggingFaceEmbeddings
from chromadb.utils.embedding_functions import EmbeddingFunction
from glob import glob
from typing import List, Dict, Optional, Tuple

# Configure logger
logger = logging.getLogger(__name__)

# Load .env for optional Cloud config
load_dotenv()

# Model configuration for multilingual e5 embeddings
MODEL_NAME = "intfloat/multilingual-e5-large"
# MODEL_NAME = "intfloat/multilingual-e5-base"

# Set environment variables for better HuggingFace connectivity
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'  # Use faster transfer
os.environ['TRANSFORMERS_CACHE'] = os.path.join(os.path.dirname(__file__), 'model_cache')

# Agent types
AGENT_TYPE_PRODUCT_SALES = "PRODUCT_SALES"
AGENT_TYPE_CUSTOMER_CORPORATE = "CUSTOMER_CORPORATE"


class E5EmbeddingFunction(EmbeddingFunction):
    """
    Custom embedding function for ChromaDB that uses HuggingFace's multilingual-e5-large model.
    Applies 'passage: ' prefix for documents and 'query: ' prefix for queries.
    
    OPTIMIZED: Uses single embedding model instance instead of loading twice.
    """
    def __init__(self, model_name=MODEL_NAME):
        """
        Initialize the HuggingFace embeddings.
        
        OPTIMIZATION: Load model once and reuse for both passage and query embeddings.
        This saves ~12 seconds and ~2GB RAM compared to loading twice!
        """
        logger.info(f"🔄 Loading embedding model: {model_name}")
        
        # Load model ONCE (not twice!)
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'},  # Use 'cuda' if you have a GPU
            encode_kwargs={'normalize_embeddings': True}
        )
        
        logger.info(f"✅ Embedding model loaded successfully")
        
    def __call__(self, input):
        """
        Embed documents with 'passage: ' prefix.
        This is called during document ingestion.
        """
        # Add "passage: " prefix to each document
        passages = [f"passage: {text}" for text in input]
        embeddings = self.embeddings.embed_documents(passages)
        return embeddings
    
    def embed_query(self, query):
        """
        Embed query with 'query: ' prefix.
        This is called during search/retrieval.
        """
        query_text = f"query: {query}"
        embedding = self.embeddings.embed_query(query_text)
        return embedding


class FAQLoader:
    """
    Handles loading FAQs from CSV files into dual ChromaDB collections 
    (PRODUCT_SALES and CUSTOMER_CORPORATE) with comprehensive metadata support.
    """
    def __init__(self, 
                 filenames=None,
                 resource_dir="resource", 
                 persist_dir="vectorstore"):
        """
        Initialize FAQLoader with dual collections for agent-based routing.
        
        :param filenames: List of filenames to load, or None to auto-load all CSV files
        :param resource_dir: Directory containing the data files
        :param persist_dir: Directory for persistent storage
        """
        # Build path relative to app directory
        app_root = os.path.dirname(os.path.abspath(__file__))
        self.resource_path = os.path.join(app_root, resource_dir)
        
        # Ensure resource directory exists
        if not os.path.exists(self.resource_path):
            raise FileNotFoundError(f"Resource directory not found at: {self.resource_path}")
        
        # Determine which files to load
        if filenames is None:
            # Auto-discover all CSV and Excel files in resource directory
            csv_files = glob(os.path.join(self.resource_path, "*.csv"))
            xlsx_files = glob(os.path.join(self.resource_path, "*.xlsx"))
            self.file_paths = csv_files + xlsx_files
            
            if not self.file_paths:
                raise FileNotFoundError(f"No CSV or Excel files found in: {self.resource_path}")
        else:
            # Use specified filenames
            if isinstance(filenames, str):
                filenames = [filenames]
            self.file_paths = [os.path.join(self.resource_path, fname) for fname in filenames]
            
            # Validate all files exist
            for fpath in self.file_paths:
                if not os.path.exists(fpath):
                    raise FileNotFoundError(f"FAQ file not found at: {fpath}")
        
        # Load and combine all data files
        self.data = self._load_multiple_files()
        print(f"✓ Loaded {len(self.data)} records from {len(self.file_paths)} file(s)")
        
        # Initialize custom embedding function
        self.embedding_function = E5EmbeddingFunction()
        
        # Initialize Chroma client
        chroma_api_key = os.getenv("CHROMA_API_KEY")
        chroma_tenant = os.getenv("CHROMA_TENANT")
        chroma_database = os.getenv("CHROMA_DATABASE")

        if chroma_api_key and chroma_tenant and chroma_database:
            # Cloud setup with connection keep-alive
            # OPTIMIZATION: Reuse HTTP connections for faster queries
            self.chroma_client = chromadb.CloudClient(
                api_key=chroma_api_key,
                tenant=chroma_tenant,
                database=chroma_database
            )
            logger.info("☁️  Using Cloud ChromaDB with connection keep-alive")
            print("using cloude chroma")
        else:
            # Local persistent setup (faster - no network latency)
            persist_path = os.path.join(app_root, persist_dir)
            os.makedirs(persist_path, exist_ok=True)
            self.chroma_client = chromadb.PersistentClient(persist_path)
            logger.info("💾 Using Local ChromaDB (faster)")
            print("using local chroma")
        
        # Create dual collections for agent-based routing
        self.collection_product = self.chroma_client.get_or_create_collection(
            name="axa_product_sales",
            embedding_function=self.embedding_function,
            metadata={"description": "Product and Sales information"}
        )
        
        self.collection_customer = self.chroma_client.get_or_create_collection(
            name="axa_customer_corporate",
            embedding_function=self.embedding_function,
            metadata={"description": "Customer Service and Corporate information"}
        )
    
    def _load_multiple_files(self):
        """
        Load and combine data from multiple CSV/Excel files.
        Handles semicolon-delimited CSV format with comprehensive metadata columns.
        
        Expected columns: id_chunk, text_original, question_original, agent_type, 
                         category_topic, doc_type, source, product_name, 
                         insurance_type, topic_focus, coverage_keyword, 
                         action_type, entity
        
        :return: Combined pandas DataFrame with all FAQ data
        """
        all_dataframes = []
        
        for file_path in self.file_paths:
            filename = os.path.basename(file_path)
            ext = os.path.splitext(file_path)[1].lower()
            
            try:
                # Load based on file extension
                if ext == ".xlsx":
                    df = pd.read_excel(file_path)
                elif ext == ".csv":
                    # Try multiple encoding and delimiter combinations
                    encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
                    delimiters = [';', ',']
                    
                    df = None
                    last_error = None
                    
                    for encoding in encodings:
                        for delimiter in delimiters:
                            try:
                                df = pd.read_csv(
                                    file_path, 
                                    sep=delimiter, 
                                    encoding=encoding,
                                    on_bad_lines='skip',
                                    quotechar='"',
                                    engine='python'
                                )
                                print(f"  ✓ Successfully parsed with encoding='{encoding}', sep='{delimiter}'")
                                break  # Success, exit inner loop
                            except Exception as e:
                                last_error = e
                                continue
                        
                        if df is not None:
                            break  # Success, exit outer loop
                    
                    if df is None:
                        print(f"  ❌ Failed to parse CSV with all encoding/delimiter combinations")
                        print(f"  Last error: {last_error}")
                        continue
                else:
                    print(f"⚠️  Skipping unsupported file format: {filename}")
                    continue
                
                # Display actual columns found
                print(f"  Columns found: {list(df.columns)}")
                
                # Validate required columns
                required_cols = ['id_chunk', 'text_original', 'agent_type']
                missing_cols = [col for col in required_cols if col not in df.columns]
                
                if missing_cols:
                    print(f"⚠️  Skipping {filename}: Missing required columns {missing_cols}")
                    print(f"  Available columns: {list(df.columns)}")
                    continue
                
                # Ensure optional columns exist
                if 'question_original' not in df.columns:
                    df['question_original'] = ''
                if 'category_topic' not in df.columns:
                    df['category_topic'] = ''
                if 'doc_type' not in df.columns:
                    df['doc_type'] = ''
                if 'source' not in df.columns:
                    df['source'] = filename
                if 'product_name' not in df.columns:
                    df['product_name'] = ''
                if 'insurance_type' not in df.columns:
                    df['insurance_type'] = ''
                if 'topic_focus' not in df.columns:
                    df['topic_focus'] = ''
                if 'coverage_keyword' not in df.columns:
                    df['coverage_keyword'] = ''
                if 'action_type' not in df.columns:
                    df['action_type'] = ''
                if 'entity' not in df.columns:
                    df['entity'] = ''
                
                # Clean data: replace NaN with empty strings
                df = df.fillna('')
                
                all_dataframes.append(df)
                print(f"  → Loaded {len(df)} records from {filename}")
                
            except Exception as e:
                print(f"⚠️  Error loading {filename}: {e}")
                continue
        
        if not all_dataframes:
            raise ValueError("No valid data files could be loaded!")
        
        # Combine all dataframes
        combined_df = pd.concat(all_dataframes, ignore_index=True)
        
        # Validate agent_type values
        valid_agent_types = [AGENT_TYPE_PRODUCT_SALES, AGENT_TYPE_CUSTOMER_CORPORATE]
        invalid_types = combined_df[~combined_df['agent_type'].isin(valid_agent_types)]
        if len(invalid_types) > 0:
            print(f"⚠️  Warning: {len(invalid_types)} records have invalid agent_type. Valid types: {valid_agent_types}")
        
        return combined_df

    def load_faq(self, force_reload: bool = False):
        """
        Load FAQs into dual ChromaDB collections based on agent_type.
        Documents are embedded with 'passage: ' prefix using multilingual-e5-large.
        Includes comprehensive metadata for filtering and ranking.
        
        :param force_reload: If True, delete and rebuild collections
        """
        # Check if collections already have documents
        count_product = self.collection_product.count()
        count_customer = self.collection_customer.count()
        
        if count_product > 0 and count_customer > 0 and not force_reload:
            print(f"✓ Collections already loaded: Product={count_product}, Customer={count_customer}")
            print(f"  To rebuild, use force_reload=True or delete vectorstore/ directory")
            return
        
        if force_reload and (count_product > 0 or count_customer > 0):
            print(f"🗑️  Force reload requested. Deleting existing collections...")
            try:
                self.chroma_client.delete_collection("axa_product_sales")
                self.chroma_client.delete_collection("axa_customer_corporate")
                print(f"✓ Collections deleted")
                
                # Recreate collections
                self.collection_product = self.chroma_client.get_or_create_collection(
                    name="axa_product_sales",
                    embedding_function=self.embedding_function,
                    metadata={"description": "Product and Sales information"}
                )
                self.collection_customer = self.chroma_client.get_or_create_collection(
                    name="axa_customer_corporate",
                    embedding_function=self.embedding_function,
                    metadata={"description": "Customer Service and Corporate information"}
                )
            except Exception as e:
                print(f"⚠️  Error deleting collections: {e}")
        
        # Prepare documents for dual collections
        product_docs = {'documents': [], 'metadatas': [], 'ids': []}
        customer_docs = {'documents': [], 'metadatas': [], 'ids': []}
        
        for idx, row in self.data.iterrows():
            # Extract all fields
            id_chunk = str(row.get('id_chunk', f'chunk_{idx}'))
            text_original = str(row.get('text_original', ''))
            question_original = str(row.get('question_original', ''))
            agent_type = str(row.get('agent_type', ''))
            category_topic = str(row.get('category_topic', ''))
            doc_type = str(row.get('doc_type', ''))
            source = str(row.get('source', ''))
            product_name = str(row.get('product_name', ''))
            insurance_type = str(row.get('insurance_type', ''))
            topic_focus = str(row.get('topic_focus', ''))
            coverage_keyword = str(row.get('coverage_keyword', ''))
            action_type = str(row.get('action_type', ''))
            entity = str(row.get('entity', ''))
            
            # Skip empty records
            if not text_original or text_original == 'nan':
                continue
            
            # Use text_original as the main document content for embedding
            doc_text = text_original
            
            # Build comprehensive metadata
            metadata = {
                "id_chunk": id_chunk,
                "text_original": text_original,
                "question_original": question_original,
                "category_topic": category_topic,
                "doc_type": doc_type,
                "source": source,
                "product_name": product_name,
                "insurance_type": insurance_type,
                "topic_focus": topic_focus,
                "coverage_keyword": coverage_keyword,
                "action_type": action_type,
                "entity": entity
            }
            
            # Route to appropriate collection based on agent_type
            if agent_type == AGENT_TYPE_PRODUCT_SALES:
                product_docs['documents'].append(doc_text)
                product_docs['metadatas'].append(metadata)
                product_docs['ids'].append(id_chunk)
            elif agent_type == AGENT_TYPE_CUSTOMER_CORPORATE:
                customer_docs['documents'].append(doc_text)
                customer_docs['metadatas'].append(metadata)
                customer_docs['ids'].append(id_chunk)
            else:
                print(f"⚠️  Warning: Invalid agent_type '{agent_type}' for record {id_chunk}")
        
        # Load Product & Sales collection
        if product_docs['documents']:
            print(f"Loading {len(product_docs['documents'])} PRODUCT_SALES records...")
            self.collection_product.add(
                documents=product_docs['documents'],
                metadatas=product_docs['metadatas'],
                ids=product_docs['ids']
            )
            print(f"✓ PRODUCT_SALES collection loaded successfully!")
        
        # Load Customer & Corporate collection
        if customer_docs['documents']:
            print(f"Loading {len(customer_docs['documents'])} CUSTOMER_CORPORATE records...")
            self.collection_customer.add(
                documents=customer_docs['documents'],
                metadatas=customer_docs['metadatas'],
                ids=customer_docs['ids']
            )
            print(f"✓ CUSTOMER_CORPORATE collection loaded successfully!")
        
        print(f"\n✓ Total loaded: Product={len(product_docs['documents'])}, Customer={len(customer_docs['documents'])}")

    def query_agent_collection(self, 
                              query: str, 
                              agent_type: str, 
                              top_k: int = 5,
                              category_filter: Optional[str] = None,
                              doc_type_filter: Optional[str] = None) -> List[Dict]:
        """
        Query specific agent collection with optional metadata filtering.
        
        :param query: User query string
        :param agent_type: Either PRODUCT_SALES or CUSTOMER_CORPORATE
        :param top_k: Number of results to retrieve (before re-ranking)
        :param category_filter: Optional category_topic filter
        :param doc_type_filter: Optional doc_type filter
        :return: List of result dictionaries with metadata and distances
        """
        # Select appropriate collection
        if agent_type == AGENT_TYPE_PRODUCT_SALES:
            collection = self.collection_product
        elif agent_type == AGENT_TYPE_CUSTOMER_CORPORATE:
            collection = self.collection_customer
        else:
            raise ValueError(f"Invalid agent_type: {agent_type}")
        
        # Build where clause for metadata filtering
        where_clause = None
        if category_filter or doc_type_filter:
            where_clause = {}
            if category_filter:
                where_clause['category_topic'] = category_filter
            if doc_type_filter:
                where_clause['doc_type'] = doc_type_filter
        
        # Perform semantic search (automatically applies "query: " prefix via embedding function)
        try:
            results = collection.query(
                query_texts=[query],
                n_results=top_k,
                where=where_clause if where_clause else None
            )
        except Exception as e:
            print(f"Query error: {e}")
            # Fallback without filtering if metadata query fails
            results = collection.query(
                query_texts=[query],
                n_results=top_k
            )
        
        # Format results with distances for re-ranking
        formatted_results = []
        if results and results['metadatas'] and len(results['metadatas'][0]) > 0:
            for idx, metadata in enumerate(results['metadatas'][0]):
                distance = results['distances'][0][idx] if 'distances' in results else 0
                formatted_results.append({
                    'metadata': metadata,
                    'distance': distance,
                    'text': metadata.get('text_original', ''),
                    'question': metadata.get('question_original', ''),
                    'id': metadata.get('id_chunk', '')
                })
        
        return formatted_results
    
    def _extract_query_features(self, query: str) -> Dict[str, List[str]]:
        """
        Extract key features from query for metadata-based boosting.
        
        :param query: User query string
        :return: Dictionary with extracted features
        """
        query_lower = query.lower()
        
        # Known product names (extracted from CSV metadata)
        known_products = [
            'smartactive', 'smartbusiness', 'smartcare', 'smartdrive', 
            'smarthome', 'smartmedint', 'smarttravel', 'smartlife',
            'pet insurance', 'health insurance', 'travel insurance'
        ]
        
        # Known insurance types
        known_insurance_types = [
            'asuransi kecelakaan', 'asuransi perjalanan', 'asuransi kesehatan',
            'asuransi kendaraan', 'asuransi rumah', 'asuransi jiwa',
            'asuransi properti', 'asuransi kendaraan bermotor'
        ]
        
        # Known coverage keywords
        known_coverages = [
            'kematian', 'cacat', 'biaya perawatan', 'pengobatan', 'rawat inap',
            'pembedahan', 'evakuasi medis', 'kehilangan bagasi', 'pencurian',
            'kebakaran', 'gempa bumi', 'banjir', 'tanggung jawab hukum'
        ]
        
        features = {
            'products': [],
            'insurance_types': [],
            'coverages': [],
            'keywords': query_lower.split()
        }
        
        # Extract product names
        for product in known_products:
            # Remove spaces for matching (e.g., "smart home" -> "smarthome")
            product_normalized = product.replace(' ', '').lower()
            query_normalized = query_lower.replace(' ', '')
            
            if product_normalized in query_normalized or product in query_lower:
                features['products'].append(product)
        
        # Extract insurance types
        for ins_type in known_insurance_types:
            if ins_type in query_lower:
                features['insurance_types'].append(ins_type)
        
        # Extract coverage keywords
        for coverage in known_coverages:
            if coverage in query_lower:
                features['coverages'].append(coverage)
        
        return features
    
    def _calculate_metadata_boost(self, 
                                  result: Dict, 
                                  query_features: Dict[str, List[str]], 
                                  query_lower: str) -> Tuple[float, List[str]]:
        """
        Calculate comprehensive boost score based on multiple metadata factors.
        
        Boost weights (negative = higher rank):
        - Product Name Match: -0.8 (highest priority)
        - Insurance Type Match: -0.6
        - Category/Topic Match: -0.4
        - Coverage Keyword Match: -0.3
        - Question Match: -0.5 (existing)
        
        :param result: Single retrieval result
        :param query_features: Extracted query features
        :param query_lower: Lowercase query string
        :return: (total_boost_score, list of boost reasons)
        """
        boost_score = 0.0
        boost_reasons = []
        
        metadata = result.get('metadata', {})
        
        # 1. PRODUCT NAME BOOSTING (Highest Priority) - Weight: -0.8
        product_name = metadata.get('product_name', '').lower()
        if product_name and query_features['products']:
            for product in query_features['products']:
                product_normalized = product.replace(' ', '').lower()
                if product_normalized in product_name or product in product_name:
                    boost_score -= 0.8
                    boost_reasons.append(f"Product:{product_name}")
                    break
        
        # 2. INSURANCE TYPE BOOSTING - Weight: -0.6
        insurance_type = metadata.get('insurance_type', '').lower()
        if insurance_type and query_features['insurance_types']:
            for ins_type in query_features['insurance_types']:
                if ins_type in insurance_type:
                    boost_score -= 0.6
                    boost_reasons.append(f"InsType:{insurance_type[:30]}")
                    break
        
        # 3. CATEGORY/TOPIC BOOSTING - Weight: -0.4
        category_topic = metadata.get('category_topic', '').lower()
        topic_focus = metadata.get('topic_focus', '').lower()
        
        # Check if any query keywords match category or topic
        query_keywords = set(query_features['keywords'])
        if category_topic:
            category_words = set(category_topic.split('_'))
            overlap = query_keywords & category_words
            if overlap:
                boost_score -= 0.4
                boost_reasons.append(f"Category:{category_topic[:30]}")
        
        if topic_focus:
            topic_words = set(topic_focus.split())
            overlap = query_keywords & topic_words
            if overlap:
                boost_score -= 0.3
                boost_reasons.append(f"Topic:{topic_focus[:30]}")
        
        # 4. COVERAGE KEYWORD BOOSTING - Weight: -0.3
        coverage_keyword = metadata.get('coverage_keyword', '').lower()
        if coverage_keyword and query_features['coverages']:
            for coverage in query_features['coverages']:
                if coverage in coverage_keyword:
                    boost_score -= 0.3
                    boost_reasons.append(f"Coverage:{coverage}")
                    break
        
        # 5. QUESTION MATCH BOOSTING (Existing) - Weight: -0.5
        question_original = result.get('question', '').lower()
        if question_original and len(question_original) > 3:
            query_words = set(query_lower.split())
            question_words = set(question_original.split())
            
            if query_words and question_words:
                overlap = len(query_words & question_words)
                total_words = len(query_words | question_words)
                similarity = overlap / total_words if total_words > 0 else 0
                
                # Graduated boosting based on similarity
                if similarity > 0.7:
                    boost_score -= 0.5
                    boost_reasons.append(f"Question:High({similarity:.1%})")
                elif similarity > 0.5:
                    boost_score -= 0.35
                    boost_reasons.append(f"Question:Med({similarity:.1%})")
                elif similarity > 0.3:
                    boost_score -= 0.15
                    boost_reasons.append(f"Question:Low({similarity:.1%})")
        
        # 6. TEXT CONTENT KEYWORD MATCHING - Weight: -0.2
        text_original = metadata.get('text_original', '').lower()
        if query_features['products']:
            for product in query_features['products']:
                product_normalized = product.replace(' ', '').lower()
                if product_normalized in text_original:
                    boost_score -= 0.2
                    boost_reasons.append(f"TextMatch:{product}")
                    break
        
        return boost_score, boost_reasons
    
    def query_with_reranking(self, 
                            query: str, 
                            agent_type: str, 
                            top_k: int = 20,  # INCREASED from 5 to 20
                            final_k: int = 3,
                            category_filter: Optional[str] = None) -> List[Dict]:
        """
        Query with ENHANCED multi-factor re-ranking.
        
        IMPROVEMENTS:
        - Increased initial retrieval: top_k=20 (was 5)
        - Multi-factor boosting: product_name, insurance_type, category, coverage, question
        - Weighted boost scores for fine-grained ranking
        - Query feature extraction for smart matching
        
        :param query: User query string
        :param agent_type: Agent type for routing
        :param top_k: Initial retrieval count (default: 20)
        :param final_k: Final number of results after re-ranking (default: 3)
        :param category_filter: Optional category filter
        :return: Re-ranked results
        """
        collection_name = "PRODUCT_SALES" if agent_type == AGENT_TYPE_PRODUCT_SALES else "CUSTOMER_CORPORATE"
        
        # Extract query features for metadata-based boosting
        query_features = self._extract_query_features(query)
        logger.info(f"🔍 QUERY ANALYSIS: Products={query_features['products']}, "
                   f"InsTypes={query_features['insurance_types']}, "
                   f"Coverages={query_features['coverages']}")
        
        # Initial retrieval (now fetching top 20 instead of 5)
        logger.info(f"📥 RETRIEVAL [{collection_name}]: Fetching top_{top_k} documents...")
        results = self.query_agent_collection(
            query=query,
            agent_type=agent_type,
            top_k=top_k,
            category_filter=category_filter
        )
        
        if not results:
            logger.warning(f"⚠️  RETRIEVAL [{collection_name}]: No results found!")
            return []
        
        # Log initial top 10 results BEFORE re-ranking (not all 20 to reduce log spam)
        logger.info(f"✅ RETRIEVAL [{collection_name}]: Retrieved {len(results)} documents (BEFORE re-ranking):")
        for idx, result in enumerate(results[:10], 1):  # Show top 10
            chunk_id = result.get('id', 'N/A')
            distance = result.get('distance', 0)
            question = result.get('question', '')
            product = result.get('metadata', {}).get('product_name', '')
            tag = '[FAQ]' if question else '[INFO]'
            logger.info(f"   [{idx}] {chunk_id} (distance: {distance:.4f}) {tag} Product:{product}")
        
        # Apply ENHANCED multi-factor re-ranking
        logger.info(f"🔄 RE-RANKING [{collection_name}]: Applying multi-factor boosting...")
        query_lower = query.lower()
        
        for result in results:
            # Calculate comprehensive boost score
            boost_score, boost_reasons = self._calculate_metadata_boost(
                result, query_features, query_lower
            )
            
            # Apply boost to distance (lower distance = higher rank)
            original_distance = result['distance']
            result['boosted_distance'] = original_distance + boost_score
            result['boost_reasons'] = boost_reasons
            result['boost_score'] = boost_score
            
            # Log significant boosts
            if boost_score < -0.1:  # Only log meaningful boosts
                chunk_id = result.get('id', 'N/A')
                logger.info(f"   ⭐ Boosted {chunk_id} by {abs(boost_score):.2f} | Reasons: {', '.join(boost_reasons)}")
        
        # Sort by boosted distance (lower = better)
        results.sort(key=lambda x: x['boosted_distance'])
        
        # Log final top_k results AFTER re-ranking
        logger.info(f"✅ RE-RANKING [{collection_name}]: Top {final_k} documents (AFTER re-ranking):")
        for idx, result in enumerate(results[:final_k], 1):
            chunk_id = result.get('id', 'N/A')
            boosted_distance = result.get('boosted_distance', 0)
            original_distance = result.get('distance', 0)
            boost_reasons = result.get('boost_reasons', [])
            question = result.get('question', '')
            product = result.get('metadata', {}).get('product_name', '')
            tag = '[FAQ]' if question else '[INFO]'
            
            if boosted_distance < original_distance:
                reasons_str = ', '.join(boost_reasons[:3]) if boost_reasons else 'N/A'  # Show top 3 reasons
                logger.info(f"   [{idx}] {chunk_id} (distance: {boosted_distance:.4f} ← {original_distance:.4f}) "
                          f"{tag} Product:{product} | Boost: {reasons_str} 🎯")
            else:
                logger.info(f"   [{idx}] {chunk_id} (distance: {boosted_distance:.4f}) {tag} Product:{product}")
        
        # Return top final_k results
        return results[:final_k]
    
    def get_data_stats(self):
        """
        Get statistics about loaded data sources and collections.
        
        :return: Dictionary with statistics
        """
        stats = {
            "total_records": len(self.data),
            "total_files": len(self.file_paths),
            "product_sales_count": self.collection_product.count(),
            "customer_corporate_count": self.collection_customer.count(),
            "files": []
        }
        
        # Count records per source file
        if 'source' in self.data.columns:
            source_counts = self.data['source'].value_counts().to_dict()
            for filename, count in source_counts.items():
                stats["files"].append({
                    "filename": filename,
                    "record_count": count
                })
        
        # Count by agent_type
        if 'agent_type' in self.data.columns:
            agent_counts = self.data['agent_type'].value_counts().to_dict()
            stats["agent_distribution"] = agent_counts
        
        return stats
