# 🤖 Financial Analyst Assistant

A sophisticated RAG (Retrieval-Augmented Generation) system that provides intelligent analysis of SEC 10-K filings for major tech companies using LangGraph, ChromaDB, and Gemini 2.5 Pro.

## 🎯 Overview

This application analyzes the latest 10-K filings from five major technology companies (Apple, Microsoft, Google, Amazon, Meta) and provides intelligent responses to financial queries. The system combines semantic search, keyword matching, intelligent caching, and conversation context to deliver accurate and relevant financial insights.

## ✨ Key Features

- **Multi-Company Analysis**: Supports AAPL, MSFT, GOOG, AMZN, META
- **Hybrid Retrieval System**: Combines ChromaDB vector search with BM25 keyword search
- **Intelligent Query Construction**: Refines user queries using conversation context
- **Smart Caching**: Redis-based caching system with semantic similarity matching
- **Reranking Pipeline**: Uses CrossEncoder for improved result relevance
- **Conversation Memory**: PostgreSQL-backed conversation history
- **Dual-Section Coverage**: Risk Factors (1A) and Management Discussion (7)
- **Interactive UI**: Clean Streamlit interface with real-time responses

## 🏗️ Architecture

### Graph Architecture
![Financial Analyst Assistant Graph](Financial%20Analyst%20Assistant%20Graph.png)

The system follows a sophisticated LangGraph workflow:

1. **Query Construction**: Analyzes user intent and conversation history to create refined, standalone queries
2. **Cache Check**: Searches Redis cache for similar previous queries (90% similarity threshold)
3. **Hybrid Retrieval**: If cache miss, performs semantic + keyword search with metadata filtering
4. **Reranking**: Uses CrossEncoder to rank and select top 2 most relevant documents
5. **Answer Generation**: LLM generates response and stores in cache for future use

### Application UI
![Financial Analyst Assistant App](Financial%20Analyst%20Assistant%20App.png)

Clean, intuitive interface with:
- Real-time chat interaction
- Conversation history persistence
- Clear conversation controls
- Loading indicators for processing states

## 🛠️ Technical Stack

- **LLM**: Google Gemini 2.5 Pro with relaxed safety settings
- **Embeddings**: BAAI/bge-small-en-v1.5 (FastEmbed)
- **Vector Database**: ChromaDB for semantic search
- **Keyword Search**: BM25Retriever for lexical matching
- **Caching**: Redis Vector Store with TTL
- **Reranking**: CrossEncoder (ms-marco-MiniLM-L-6-v2)
- **Orchestration**: LangGraph for workflow management
- **Persistence**: PostgreSQL for conversation checkpoints
- **Frontend**: Streamlit for interactive UI
- **Data Source**: SEC API for 10-K filings

## 📋 Prerequisites

- Python 3.8+
- Docker (for Redis)
- PostgreSQL database
- Google Gemini API key
- SEC API key

## 🚀 Installation

1. **Clone the repository**
```bash
git clone https://github.com/eslammohamedtolba/Financial-Analyst-Assistant.git
cd Financial-Analyst-Assistant
```

2. **Create and activate virtual environment**
```bash
# Create the environment
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on macOS/Linux
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**
Create a `.env` file with:
```env
# Google Gemini API
GOOGLE_API_KEY=your_gemini_api_key

# SEC API
SEC_API_KEY=your_sec_api_key

# Database Configuration
DB_USER=myuser
DB_PASSWORD=mypassword
DB_HOST=localhost
DB_PORT=5432
DB_NAME=graph_memory
```

5. **Start Docker containers**

**PostgreSQL container:**
```bash
docker run \
  --name my-postgres \
  -d \
  -p 5432:5432 \
  -e POSTGRES_USER=myuser \
  -e POSTGRES_PASSWORD=mypassword \
  -e POSTGRES_DB=graph_memory \
  -v postgres_data:/var/lib/postgresql/data \
  postgres:16
```

**Redis container:**
```bash
docker run \
  --name my-redis \
  -p 6379:6379 \
  -v redis-data:/data \
  -d \
  redis/redis-stack:latest
```

6. **Process financial data**
Run the Jupyter notebook to download and process SEC filings:
```bash
jupyter notebook "Financial Data Analysis Preparation.ipynb"
```

7. **Launch the application**
```bash
streamlit run main.py
```

## 📁 Project Structure

```
financial-analyst-assistant/
├── main.py                 # Streamlit application entry point
├── graph.py               # LangGraph workflow definition
├── nodes.py              # Individual processing nodes
├── state.py              # State management and Pydantic models
├── config.py             # Configuration and model initialization
├── db_utils.py           # Database utility functions
├── requirements.txt      # Python dependencies
├── .env                 # Environment variables
├── Data/
│   ├── chroma_db/       # ChromaDB vector store
│   ├── bm25_retriever.pkl # Serialized BM25 retriever
│   └── Financial Data Analysis Preparation.ipynb
├── Financial Analyst Assistant App.png
├── Financial Analyst Assistant Graph.png
└── README.md
```

## 🔧 Core Components

### Query Construction
- Analyzes conversation context to resolve pronouns and follow-up questions
- Extracts metadata filters (company ticker, document category)
- Creates standalone, retrieval-optimized queries

### Hybrid Retrieval System
- **Semantic Search**: ChromaDB with metadata filtering
- **Keyword Search**: BM25 for exact term matching
- **Deduplication**: Removes duplicate results across retrievers
- **Reranking**: CrossEncoder scores and selects top documents

### Intelligent Caching
- Redis-based semantic similarity caching
- 90% similarity threshold for cache hits
- 24-hour TTL for cached responses
- Reduces API calls and improves response times

### Conversation Management
- PostgreSQL checkpointer for conversation persistence
- Thread-based session management
- Conversation history display and clearing functionality

## 📊 Supported Companies

| Ticker | Company Name |
|--------|-------------|
| AAPL | Apple Inc. |
| MSFT | Microsoft Corporation |
| GOOG | Alphabet Inc. (Google) |
| AMZN | Amazon.com, Inc. |
| META | Meta Platforms, Inc. |

## 🎯 Document Categories

- **risks**: Risk Factors (Section 1A of 10-K filings)
- **management_dis**: Management's Discussion and Analysis (Section 7 of 10-K filings)

## 💡 Usage Examples

- "What are the main risks for Apple?"
- "Tell me about Microsoft's revenue trends"
- "How does Google's management view their competitive position?"
- "What challenges does Amazon face in their cloud business?"
- "Compare the risk factors between Meta and Apple"

## 🔒 Safety & Compliance

- Relaxed safety settings on Gemini for financial document analysis
- Proper error handling and fallback mechanisms
- Respect for API rate limits and usage guidelines
- Secure environment variable management

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request
