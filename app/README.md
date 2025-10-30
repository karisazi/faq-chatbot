# AXA FAQ Chatbot

Multilingual RAG chatbot built with Streamlit, Gemini, E5 embeddings, and ChromaDB. It routes queries to the right domain, retrieves relevant knowledge, and generates grounded answers with source attribution.

## Quick Start

1) Install dependencies
```bash
pip install -r requirements.txt
```

2) Configure environment
Create `.env` in this folder:
```env
GEMINI_API_KEY=your_gemini_api_key_here
# Optional for Chroma Cloud; omit to use local storage
# CHROMA_API_KEY=
# CHROMA_TENANT=
# CHROMA_DATABASE=
```

3) Verify your data file
```bash
cd testing
python check_csv.py
```

4) Run the app
```bash
streamlit run main.py
```

## Data Format
Place your knowledge base at `resource/AXA_QNA.csv` with UTF‑8 and semicolon delimiter.

Minimum required columns:
```csv
id_chunk;text_original;agent_type
```

Recommended additional columns (improves ranking and filtering):
`question_original, category_topic, doc_type, source, product_name, insurance_type, topic_focus, coverage_keyword, action_type, entity`

Agent types must be one of: `PRODUCT_SALES` or `CUSTOMER_CORPORATE`.

## Features
- Dual collections in ChromaDB for product/sales and customer/corporate
- Supervisor routing + specialist answering
- Multilingual E5 embeddings with query/passages prefixing
- Metadata-aware re-ranking (product, insurance type, category, coverage, question, text)
- Streamlit UI with cached initialization and simple query cache

## Configuration
- Retrieval size: edit `faq_loader.query_with_reranking(top_k=20, final_k=3)` in `chains.py`
- Model: set `GEMINI_MODEL` via `.env` (defaults to `gemini-2.0-flash`)
- GPU for embeddings: in `faq_loader.py` set `model_kwargs={'device':'cuda'}` (if available)
- Local vs Cloud ChromaDB: provide CHROMA_* vars to use Cloud; otherwise uses local `vectorstore/`

## Testing
```bash
# CSV diagnostics
cd testing && python check_csv.py

# End-to-end system test
python test_system.py

# Boosting behavior
python test_boosting.py
```

## Troubleshooting
- CSV parsing errors: run `testing/check_csv.py` and ensure semicolon `;` delimiter and consistent columns
- Slow first run: initial E5 model download (~2GB) is expected; subsequent queries are faster
- Repeated model loads: clear Streamlit cache
```bash
streamlit cache clear
```
- Rebuild local vectorstore
```bash
rm -rf vectorstore/
```

## Project Structure
```text
app/
├── main.py            # Streamlit UI
├── chains.py          # Supervisor + specialists + orchestration
├── faq_loader.py      # Data loading, ChromaDB, embeddings, re-ranking
├── utils.py           # Input validation and normalization
├── resource/          # AXA_QNA.csv lives here
├── testing/           # Diagnostic and system tests
├── styles.css         # UI styles
└── requirements.txt   # Dependencies
```

## Requirements
Python 3.10+ and the packages listed in `requirements.txt`.

## Security Notes
- Validates inputs and avoids prompt injection patterns
- Answers grounded in retrieved context; provides fallbacks when context is missing

## License
Proprietary. Do not distribute without permission.

