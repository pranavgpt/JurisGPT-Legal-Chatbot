# 🏛 JurisGPT - A Legal ChatBot

An AI-powered chatbot designed to provide intelligent legal information retrieval and decision support.  
Built with **Python**, this project processes legal documents, performs semantic search, and answers domain-specific questions efficiently.

---

## Introduction

JurisGPT aims to assist users by providing accurate and concise legal information based on the Indian Penal Code and related legal documents. The chatbot retrieves relevant context from the knowledge base to answer user queries efficiently.

---

## Features

- **Document Ingestion** – Reads and processes legal documents for easy querying.
- **Semantic Search** – Finds relevant clauses, sections, and references.
- **Decision Engine** – Suggests relevant legal interpretations based on input queries.
- **Configurable & Scalable** – Works with multiple document types (PDF, DOCX, emails).

---

## Architecture

The architecture of JurisGPT includes the following components:

1. **Document Loader:** Loads legal documents from a directory of PDF files.  
2. **Text Splitter:** Splits documents into manageable chunks for embedding.  
3. **Embeddings:** Uses Google Generative AI Embeddings to transform text into vector representations.  
4. **Vector Store:** Utilizes FAISS to store and retrieve document embeddings.  
5. **LLM:** Uses the ChatGroq API to generate responses based on retrieved documents and user queries.  
6. **Memory:** Maintains a conversation buffer to provide context in conversations.  

---

## Setup and Installation

### Prerequisites
- Python 3.12  
- Streamlit  
- LangChain Community  
- Google Generative AI  
- FAISS  

---

### Installation Steps

**1. Clone the Repository**
```bash
git clone https://github.com/pranavgpt/JurisGPT-Legal-Chatbot.git
cd JurisGPT

```

**2. Set Up and Activate Virtual Environment**
```bash
conda create -p venv python==3.12
conda activate C:\directory\venv
```

**3. Install Dependencies**
```bash
pip install -r requirements.txt
```

**4. Set Up Environment Variables**
```bash
Create a .env file in the project root directory and add your API keys:
GOOGLE_API_KEY=your_google_api_key
GROQ_API_KEY=your_groq_api_key
```
**5. Split, Embed and Save Documents**
```bash
python ingestion.py
```
---
### Usage
**Run the Streamlit application:**
```bash
streamlit run app.py
```

