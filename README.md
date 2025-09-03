# 🤖 Financial Insight Engine

A sophisticated, production-ready RAG (Retrieval-Augmented Generation) system powered by a hybrid LLM architecture. This application features a locally-run, fine-tuned **Phi-3 3.8B** for expert financial answer generation, **Google's Gemini Pro** for advanced query analysis, and **LangSmith** for full observability and monitoring.

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

-   **Hybrid LLM Architecture**: Leverages the strengths of a specialized local model (Phi-3) for generation and a powerful API model (Gemini) for query analysis.
-   **Full Observability with LangSmith**: Integrates LangSmith for end-to-end tracing of the RAG pipeline, enabling detailed monitoring, debugging, and performance analysis of each component.
-   **Fine-Tuned for Finance**: Utilizes a `microsoft/Phi-3-mini-4k-instruct` model fine-tuned on financial Q&A, providing expert, context-aware responses.
-   **High-Performance Local Inference**: Optimized with **Unsloth** for 2x faster, low-memory 4-bit inference on consumer GPUs.
-   **Hybrid Retrieval System**: Combines ChromaDB vector search with BM25 keyword search for robust document retrieval.
-   **Intelligent Query Construction**: Refines user queries using conversation context.
-   **Smart Caching**: Redis-based semantic similarity caching to reduce latency and redundant processing.
-   **Conversation Memory**: Persists conversation history using a PostgreSQL checkpointer.
-   **Production-Ready Database Pooling**: Implements **PgBouncer** to manage database connections efficiently, preventing bottlenecks under high user load.
-   **Interactive UI**: Built with Streamlit for a clean, real-time user experience.

## 📊 Model Evaluation & Results

To quantitatively validate the effectiveness of the fine-tuning process, the `RAG_Pipeline_Evaluation.ipynb` notebook was created. This notebook uses the **Ragas** framework to evaluate and compare the performance of the fine-tuned Phi-3 model against the base model on a "golden dataset" of financial questions.

The evaluation clearly demonstrates the value of fine-tuning, showing significant improvements in key areas like factual consistency (`faithfulness`).

### Evaluation Summary

| Metric             | Fine-Tuned Model | Base Model | Improvement (%) |
| ------------------ | ---------------- | ---------- | --------------- |
| **faithfulness** | 0.94583          | 0.80715    | +17.18%         |
| **answer_correctness** | 0.54475          | 0.59113    | -7.85%          |
| **answer_relevancy** | 0.96889          | 0.96889    | 0.00%           |
| **context_precision** | 1.00000          | 0.91667    | +9.09%          |
| **context_recall** | 0.44444          | 0.44444    | 0.00%           |


## 🏗️ Architecture

### The Hybrid LLM Strategy

This project employs two different LLMs, each assigned to the task it performs best:

1.  **Google Gemini Pro (Query Analysis)**: Used for its superior reasoning and structured output capabilities. It deconstructs user intent, handles conversation history to resolve follow-up questions, and extracts metadata filters for the retrieval system.
2.  **Fine-Tuned Phi-3 (Answer Generation)**: Used for the final, context-grounded response. As a specialized model, it delivers answers in the precise tone and format it was trained on, running efficiently and privately on local hardware.

### Graph Architecture

![Financial Analyst Assistant Graph](Financial%20Analyst%20Assistant%20Graph.png)

The system follows a sophisticated LangGraph workflow:
1.  **Query Construction**: Gemini analyzes user intent and conversation history.
2.  **Cache Check**: Searches Redis cache for similar previous queries.
3.  **Hybrid Retrieval**: If cache miss, performs semantic + keyword search.
4.  **Reranking**: A CrossEncoder model ranks and selects the top 2 most relevant documents.
5.  **Answer Generation**: The fine-tuned Phi-3 model generates the final response, which is then stored in the cache.

## 🛠️ Technical Stack

-   **Answer Generation LLM**: Fine-tuned `microsoft/Phi-3-mini-4k-instruct`
-   **Query Analysis LLM**: Google Gemini 1.5 Pro
-   **Fine-Tuning & Inference Engine**: Unsloth
-   **Monitoring & Observability**: LangSmith
-   **ML Framework**: PyTorch
-   **Embeddings**: `BAAI/bge-small-en-v1.5` (FastEmbed)
-   **Vector Database**: ChromaDB
-   **Keyword Search**: `rank_bm25`
-   **Caching**: Redis Vector Store
-   **Reranking**: `cross-encoder/ms-marco-MiniLM-L-6-v2`
-   **Orchestration**: LangGraph
-   **Persistence & Pooling**: PostgreSQL, PgBouncer
-   **Frontend**: Streamlit
-   **Evaluation**: Ragas

## 📋 Prerequisites

-   **Hardware**: An **NVIDIA GPU** with CUDA 12.1+ support.
    -   **Minimum 6GB VRAM** is required to run the application.
    -   8GB+ VRAM is recommended for a smoother experience.
-   **Python**: Python 3.11 (64-bit).
-   **Docker**: For running the PostgreSQL and Redis databases.
-   **API Keys & Secrets**: You will need several API keys for different parts of the project.

## 🚀 Installation & Setup

1.  **Clone the repository**
    ```bash
    git clone [https://github.com/eslammohamedtolba/Financial-Insight-Engine.git](https://github.com/eslammohamedtolba/Financial-Insight-Engine.git)
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

4.  **Set up API Keys and Environment Variables**

    This project requires two sets of credentials: one for the main Streamlit application and another for running the data processing and evaluation notebooks.

    **A) For the Streamlit Application (`.env` file):**
    Create a `.env` file in the root directory for the application's runtime credentials.
    ```env
    # Google Gemini API for the application
    GOOGLE_API_KEY="your_gemini_api_key"

    # LangSmith Configuration for Observability
    LANGSMITH_TRACING="true"
    LANGSMITH_ENDPOINT="[https://api.smith.langchain.com](https://api.smith.langchain.com)"
    LANGSMITH_API_KEY="your_langsmith_api_key"
    LANGSMITH_PROJECT="pr-pertinent-emission-45" # Or your preferred project name

    # Database Configuration
    DB_USER="myuser"
    DB_PASSWORD="mypassword"
    DB_HOST="localhost"
    DB_PORT="6432" # Connect to the pgbouncer not postgresql directly
    DB_NAME="graph_memory"
    ```

    **B) For the Jupyter Notebooks (Secrets Management):**
    The notebooks for data preparation, fine-tuning, and evaluation require additional keys. **Do not add these to the `.env` file.** Instead, manage them using a secrets manager appropriate for your environment (e.g., Colab Secrets, Kaggle Secrets, or environment variables in your cloud IDE).
    -   `OPENROUTER_API_KEY`: For accessing various models for evaluation via OpenRouter.
    -   `HF_TOKEN`: Your Hugging Face token for downloading models.
    -   `SEC_API_KEY`: Your API key for the SEC EDGAR database to download filings.
    -   `WANDB_API_KEY`: (Optional) Your Weights & Biases key for logging fine-tuning runs.


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

6.  **Run the Jupyter Notebooks (CRITICAL STEP)**
    You must run the provided Jupyter Notebooks in order. This process downloads financial data, builds the retrieval databases, fine-tunes the local model, and evaluates its performance.

    -   **1. Prepare Retrieval Databases:**
        Open and run all cells in `Knowledge_Base_Construction.ipynb`. This will create the ChromaDB vector store and the BM25 retriever file.

    -   **2. Fine-Tune the Language Model:**
        Open and run all cells in `Fine-Tuning_Phi-3_for_Financial_QA_with_Unsloth.ipynb`. This will create the `Data/phi3_finetuned_model` directory, which is required by the application.
        
    -   **3. Evaluate the RAG Pipeline:**
        Open and run all cells in `RAG_Pipeline_Evaluation.ipynb`. This will generate a quantitative comparison of the fine-tuned model versus the base model.

7.  **Launch the application**
    Ensure all setup steps are complete and your Docker containers are running.

    ```bash
    streamlit run main.py
    ```

## 📁 Project Structure
````
├── Data/
│   ├── chroma_db/                        # (Generated) ChromaDB vector store
│   ├── phi3_finetuned_model/             # (Generated) Fine-tuned Phi-3 model weights
│   ├── bm25_retriever.pkl                # (Generated) BM25 retriever for keyword search
│   ├── Knowledge_Base_Construction.ipynb                       # Notebook to create the vector & keyword databases
│   ├── Fine-Tuning_Phi-3_for_Financial_QA_with_Unsloth.ipynb   # Notebook to fine-tune the Phi-3 model
│   └── RAG_Pipeline_Evaluation.ipynb                           # Notebook to evaluate model performance with Ragas
├── .env                             # Local environment variables for the Streamlit app
├── .gitignore                       # Git ignore file
├── config.py                        # Configuration and model initialization
├── db_utils.py                     # Database utility functions
├── graph.py                         # LangGraph workflow definition
├── main.py                          # Streamlit application entry point
├── nodes.py                         # Individual processing nodes
├── README.md                        # Project README file
├── requirements.txt                 # Python dependencies
├── state.py                         # State management and Pydantic models
├── Financial Analyst Assistant App.png     # UI Screenshot
└── Financial Analyst Assistant Graph.png   # Graph Screenshot
````

## 🔒 Safety & Compliance

-   Relaxed safety settings on Gemini for financial document analysis.
-   Proper error handling and fallback mechanisms.
-   Secure environment variable management using `.env` files and secrets management.

## 🤝 Contributing

1.  Fork the repository
2.  Create a feature branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4.  Push to the branch (`git push origin feature/AmazingFeature`)
5.  Open a Pull Request
