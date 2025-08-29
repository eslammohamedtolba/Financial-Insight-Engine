# 🤖 Financial Insight Engine

A sophisticated RAG (Retrieval-Augmented Generation) system powered by a hybrid LLM architecture. This application features a locally-run, fine-tuned **Phi-3 3.8B** for expert financial answer generation and **Google's Gemini Pro** for advanced query analysis of SEC 10-K filings.

This application provides intelligent analysis of the latest 10-K filings from five major technology companies (Apple, Microsoft, Google, Amazon, Meta). The system leverages a state-of-the-art hybrid LLM strategy, combining a specialized local model for nuanced answer generation with a powerful API model for complex reasoning. By using a fine-tuned model, the assistant delivers responses with a consistent, expert tone, running efficiently on local hardware.

## 🖥️ User Interface

The Financial Insight Engine features a clean, intuitive Streamlit interface designed for seamless interaction with complex financial data:

![Financial Analyst Assistant Interface](Financial%20Analyst%20Assistant%20App.png)

The interface provides:
- **Real-time Chat**: Conversational interface for natural language queries about financial data.
- **Context-Aware Responses**: The system maintains conversation history for follow-up questions.
- **Professional Formatting**: Clean, readable responses with proper financial terminology.
- **Session Management**: Persistent conversation history throughout your analysis session.

## ✨ Key Features

-   **Hybrid LLM Architecture**: Leverages the strengths of a specialized local model (Phi-3) for generation and a powerful API model (Gemini) for query analysis.
-   **Fine-Tuned for Finance**: Utilizes a `microsoft/Phi-3-mini-4k-instruct` model fine-tuned on financial Q&A, providing expert, context-aware responses.
-   **High-Performance Local Inference**: Optimized with **Unsloth** for 2x faster, low-memory 4-bit inference on consumer GPUs.
-   **Hybrid Retrieval System**: Combines ChromaDB vector search with BM25 keyword search for robust document retrieval.
-   **Intelligent Query Construction**: Refines user queries using conversation context.
-   **Smart Caching**: Redis-based semantic similarity caching to reduce latency and redundant processing.
-   **Conversation Memory**: Persists conversation history using a PostgreSQL checkpointer.
-   **Production-Ready Database Pooling**: Implements **PgBouncer** to manage database connections efficiently, preventing bottlenecks under high user load.
-   **Interactive UI**: Built with Streamlit for a clean, real-time user experience.

## 🏗️ Architecture

### The Hybrid LLM Strategy

This project employs two different LLMs, each assigned to the task it performs best:

1.  **Google Gemini Pro (Query Analysis)**: Used for its superior reasoning and structured output capabilities. It deconstructs user intent, handles conversation history to resolve follow-up questions, and extracts metadata filters for the retrieval system.
2.  **Fine-Tuned Phi-3 (Answer Generation)**: Used for the final, context-grounded response. As a specialized model, it delivers answers in the precise tone and format it was trained on, running efficiently and privately on local hardware.

### Graph Architecture

![Financial Analyst Assistant Graph](Financial%20Analyst%20Assistant%20Graph.png)

The system follows a sophisticated LangGraph workflow:
1.  **Query Construction**: Gemini analyzes user intent and conversation history.
2.  **Cache Check**: Searches Redis cache for similar previous queries.
3.  **Hybrid Retrieval**: If cache miss, performs semantic + keyword search.
4.  **Reranking**: A CrossEncoder model ranks and selects the top 2 most relevant documents.
5.  **Answer Generation**: The fine-tuned Phi-3 model generates the final response, which is then stored in the cache.

## 🛠️ Technical Stack

-   **Answer Generation LLM**: Fine-tuned `microsoft/Phi-3-mini-4k-instruct`
-   **Query Analysis LLM**: Google Gemini 1.5 Pro
-   **Fine-Tuning & Inference Engine**: Unsloth
-   **ML Framework**: PyTorch
-   **Embeddings**: `BAAI/bge-small-en-v1.5` (FastEmbed)
-   **Vector Database**: ChromaDB
-   **Keyword Search**: `rank_bm25`
-   **Caching**: Redis Vector Store
-   **Reranking**: `cross-encoder/ms-marco-MiniLM-L-6-v2`
-   **Orchestration**: LangGraph
-   **Persistence & Pooling**: PostgreSQL, PgBouncer
-   **Frontend**: Streamlit

## 📋 Prerequisites

-   **Hardware**: An **NVIDIA GPU** with CUDA 12.1+ support.
    - **Minimum 6GB VRAM** is required to run the application.
    - 8GB+ VRAM is recommended for a smoother experience.
-   **Python**: Python 3.11 (64-bit).
-   **Docker**: For running the PostgreSQL and Redis databases.
-   **API Keys**: A Google Gemini API key.

## 🚀 Installation & Setup

1.  **Clone the repository**
    ```bash
    git clone https://github.com/eslammohamedtolba/Financial-Insight-Engine.git
    cd Financial-Insight-Engine
    ```

2.  **Create and activate virtual environment**
    ```bash
    # Create the environment
    python -m venv venv

    # Activate on Windows
    .\venv\Scripts\activate

    # Activate on macOS/Linux
    source venv/bin/activate
    ```

3.  **Install Dependencies**
    The `requirements.txt` file is configured to install all packages, including the correct CUDA version of PyTorch.
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up environment variables**

    Create a `.env` file in the root directory and add your credentials.
    ```env
    # Google Gemini API
    GOOGLE_API_KEY="your_gemini_api_key"

    # Database Configuration
    DB_USER="myuser"
    DB_PASSWORD="mypassword"
    DB_HOST="localhost"
    DB_PORT="6432" # Connect to PgBouncer, not directly to Postgres
    DB_NAME="graph_memory"
    ```



5.  **Start Docker Infrastructure**

    First, create a dedicated Docker network for the services to communicate with each other.

    ```bash
    docker network create financial-analyst-net
    ```

    Next, run the following commands to start the PostgreSQL, Redis, and PgBouncer services on the new network.

    **PostgreSQL:**
    (The backend database for storing conversation history)

    ```bash
    docker run --name my-postgres --network financial-analyst-net -d -p 5432:5432 -e POSTGRES_USER=myuser -e POSTGRES_PASSWORD=mypassword -e POSTGRES_DB=graph_memory -v postgres_data:/var/lib/postgresql/data postgres:16
    ```

    **Redis:**
    (The semantic cache for storing query results)

    ```bash
    docker run --name my-redis --network financial-analyst-net -d -p 6379:6379 -v redis-data:/data redis/redis-stack:latest
    ```

    **PgBouncer:**
    (The connection pooler that sits in front of PostgreSQL)

    ```bash
    docker run \
      --name my-pgbouncer \
      --network financial-analyst-net \
      -d \
      -p 6432:6432 \
      -e POSTGRESQL_HOST=my-postgres \
      -e POSTGRESQL_PORT=5432 \
      -e POSTGRESQL_USERNAME=myuser \
      -e POSTGRESQL_PASSWORD=mypassword \
      -e POSTGRESQL_DATABASE=graph_memory \
      -e PGBOUNCER_DATABASE=graph_memory \
      -e PGBOUNCER_USERNAME=myuser \
      -e PGBOUNCER_PASSWORD=mypassword \
      -e PGBOUNCER_PORT=6432 \
      -e PGBOUNCER_POOL_MODE=transaction \
      -e PGBOUNCER_DEFAULT_POOL_SIZE=20 \
      -e PGBOUNCER_MAX_CLIENT_CONN=1000 \
      -e PGBOUNCER_MAX_DB_CONN=100 \
      -e PGBOUNCER_AUTH_TYPE=md5 \
      bitnami/pgbouncer:latest
    ```

    *Note: `POSTGRESQL_HOST` is set to `my-postgres`, the container name, which is possible because they share a Docker network.*

6.  **Prepare Data and Fine-Tune Model (CRITICAL STEP)**
    You must run the provided Jupyter Notebooks in order. This process downloads financial data, builds the retrieval databases, and creates the local language model.

    - **Prepare Retrieval Databases:**
        Open and run all cells in `Data/Financial Data Analysis Preparation.ipynb`. This will create the ChromaDB vector store and the BM25 retriever file.

    - **Fine-Tune the Language Model:**
        Open and run all cells in `Data/Fine-Tuning_Phi-3_for_Financial_QA_with_Unsloth.ipynb`. This will create the `Data/phi3_finetuned_model` directory, which is required by the application.

7.  **Launch the application**
    Ensure all setup steps are complete and your Docker containers are running.

    ```bash
    streamlit run main.py
    ```

## 📁 Project Structure
````

├── Data/
│   ├── chroma_db/                  # (Generated) ChromaDB vector store for semantic search
│   ├── phi3_finetuned_model/      # (Generated) Fine-tuned Phi-3 model weights
│   ├── bm25_retriever.pkl          # (Generated) BM25 retriever for keyword search
│   ├── Financial Data Analysis Preparation.ipynb                # Notebook to create the vector & keyword databases
│   └── Fine-Tuning_Phi-3_for_Financial_QA_with_Unsloth.ipynb    # Notebook to fine-tune and save the Phi-3 model
├── .env                          # Local environment variables
├── .gitignore                    # Git ignore file
├── config.py                     # Configuration and model initialization
├── db_utils.py                   # Database utility functions
├── graph.py                      # LangGraph workflow definition
├── main.py                       # Streamlit application entry point
├── nodes.py                      # Individual processing nodes
├── README.md                     # Project README file
├── requirements.txt              # Python dependencies
├── state.py                      # State management and Pydantic models
├── Financial Analyst Assistant App.png      # UI Screenshot
└── Financial Analyst Assistant Graph.png    # Graph Screenshot

````

## 🔒 Safety & Compliance

-   Relaxed safety settings on Gemini for financial document analysis.
-   Proper error handling and fallback mechanisms.
-   Secure environment variable management using `.env` files.

## 🤝 Contributing

1.  Fork the repository
2.  Create a feature branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4.  Push to the branch (`git push origin feature/AmazingFeature`)
5.  Open a Pull Request