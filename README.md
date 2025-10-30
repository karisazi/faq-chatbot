---
title: "FAQ Chatbot"
emoji: "🤖"
colorFrom: "green"
colorTo: "blue"
sdk: "streamlit"
sdk_version: "1.48.1"
app_file: "app/main.py"
pinned: false
---

# 🤖 AXA Insurance FAQ Chatbot

An intelligent chatbot designed to answer frequently asked questions about AXA Insurance services, making it easy for customers to find information about premium payments, claims, policy information, and insurance products quickly in an interactive way.


## 📑 Table of Contents
  <ol>
    <li><a href="#introduction">Introduction</a></li>
    <li><a href="#demo">Demo</a></li>
    <li><a href="#features">Features</a></li>
    <li><a href="#dataset">Dataset</a></li>
    <li><a href="#model-approach">Model / Approach</a></li>
    <li><a href="#installation-setup">Installation / Setup</a></li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#results">Results</a></li>
    <li><a href="#project-structure">Project Structure</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>


## Introduction
Accessing accurate information quickly is essential, especially when users have many questions about a service or product. A key challenge is providing fast and consistent answers to **frequently asked questions (FAQs)**.

This project introduces an **intelligent FAQ chatbot** that delivers context-aware responses through natural conversations, helping users find information quickly and interactively.

By automating FAQ responses, the chatbot improves user experience, reduces manual support efforts, and ensures information is always available on demand.


## Demo
<img src="demo.gif" width="800">  

*(Example: Chatbot running on a Streamlit web app)*

⚠️ Note: This chatbot is specifically designed for AXA Insurance Indonesia and covers topics including: **Premium Payments, Claims Submission, Policy Information, and Insurance Product FAQs** for both PT AXA Financial Indonesia and PT AXA Insurance Indonesia.





## Features
- Provides **instant answers** to user queries in real-time.  
- Trained on an **FAQ and domain-specific dataset** for high accuracy.  
- Supports **context-aware conversations** using embeddings.  
- Utilizes **top_k=2** for varied responses, achieving up to **95% accuracy**.  
- Offers an **interactive Streamlit interface** for chatting with the bot.  
- **Containerized with Docker** for lightweight deployment.  



## Dataset
- Source: AXA Insurance Indonesia FAQ Dataset  
- Format: CSV file with Question and Answer pairs covering insurance topics
- Topics: Premium payments, claims submission, policy information, insurance products



## Method
- **Type**: Retrieval-Augmented Generation (RAG) Chatbot.  
- **Components**:  
  - **Embeddings**: ChromaDB with SentenceTransformer (`all-MiniLM-L6-v2`).  
  - **LLM**: Google Gemini via LangChain for intelligent responses.  
  - **Retriever**: Semantic similarity search (`top_k=2`).  
- **Techniques**:  
  - Exception handling, input validation, and security measures.  
  - Docker-based deployment for scalability and production readiness.  
- **Performance**:  
  - Achieved **95% response accuracy** on FAQ dataset.  
  - Provides context-aware answers through RAG pipeline.  



## Installation / Setup

### Prerequisites

* Python 3.10 or 3.11
* [Google Gemini API Key](https://ai.google.dev/) (for running Gemini)
* Docker (optional, for containerized deployment)

### Steps

```bash
git clone https://github.com/username/faq-chatbot.git
cd faq-chatbot
pip install -r requirements.txt

# Create .env file with your Gemini API key
echo "GEMINI_API_KEY=your_api_key_here" > .env
echo "GEMINI_MODEL_NAME=gemini-1.5-flash" >> .env
```

If you use **Docker**:

```bash
cd faq-chatbot
docker build -t chatbot .
docker run -p 8501:8501 --env-file .env chatbot
```


## Usage

### Run the chatbot locally

```bash
streamlit run app/main.py
```

### Example Interaction

```
User: Bagaimana cara membayar premi AXA?  
Bot: Pembayaran premi dapat dilakukan melalui beberapa metode:
- Autodebet dari kartu kredit (untuk jalur Telemarketing & e-Commerce)
- Autodebet rekening bank (untuk jalur Agency)
- Virtual Account
- Transfer ke rekening Bank Mandiri

User: Kapan bisa mengajukan klaim?  
Bot: Nasabah bisa mengajukan klaim untuk penyakit yang baru diderita setelah melewati masa tunggu 30 hari sejak tanggal efektif polis. Untuk kecelakaan tidak berlaku masa tunggu.
```


## Results

* Achieved **95% accuracy** in understanding user intents.
* Generates **context-aware responses** using **RAG (Retrieval-Augmented Generation)**.
* Handles multiple user queries while maintaining conversational context.




## Project Structure

```
├── app/                        # Source code for the chatbot
│   ├── AXA_QNA.csv             # AXA Insurance FAQ Dataset
│   ├── Dockerfile               # Docker setup
│   ├── chains.py                # Logic for RAG/response chains (Gemini integration)
│   ├── faq_loader.py            # Loads and processes FAQ dataset
│   ├── main.py                  # Entry point for the Streamlit app
│   ├── requirements.txt         # App-specific dependencies
│   ├── styles.css               # Custom CSS for the app
│   └── utils.py                 # Helper functions
├── test_integration.py          # Integration test script
├── chatbot_optimization.ipynb   # Notebook showing improvements (accuracy, exception handling, security)
├── requirements.txt             # Global dependencies
└── README.md                    # Project documentation

```

## Acknowledgments

* Developed using [Google Gemini](https://ai.google.dev/).
* Thanks to **Google** for providing the **Gemini API**, used with **LangChain** for context-aware responses.
* Special thanks to **AXA Insurance Indonesia** for providing the comprehensive FAQ dataset.


