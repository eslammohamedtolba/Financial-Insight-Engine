# 🤖 Financial Analyst Assistant

A production-ready, enterprise-grade RAG (Retrieval-Augmented Generation) system built with modern microservices architecture. This application features a locally-run, fine-tuned **Phi-3 3.8B** model for expert financial analysis, **Google's Gemini Pro** for intelligent query processing, and comprehensive observability through **LangSmith** integration.

This sophisticated system provides intelligent analysis of the latest 10-K filings from five major technology companies (Apple, Microsoft, Google, Amazon, Meta). The architecture implements a hybrid LLM strategy, combining a specialized local model for nuanced financial insights with a powerful cloud model for complex reasoning tasks, all deployed through a scalable FastAPI backend with multi-user authentication and conversation management.

## 🖥️ User Interface

The Financial Analyst Assistant includes a web interface for user interaction:

![Financial Analyst Assistant Interface](Financial%20Analyst%20Assistant%20App.png)

The interface provides secure user authentication, multi-conversation support, and real-time chat capabilities for financial analysis queries.

## ✨ Key Features

- **Microservices Architecture**: Decoupled FastAPI backend with domain-driven design, JWT authentication, and multi-user support
- **Hybrid LLM Strategy**: Fine-tuned Phi-3 (3.8B) for financial generation + Gemini Pro for query analysis and reasoning
- **Advanced RAG Pipeline**: LangGraph workflow orchestration with hybrid retrieval (ChromaDB + BM25), CrossEncoder reranking, and semantic caching
- **Production Infrastructure**: Docker Compose orchestration with PostgreSQL, Redis, PgBouncer connection pooling, and GPU acceleration
- **Thread-Safe Operations**: Concurrent request handling with GPU resource locking for optimal multi-user performance
- **Observability**: LangSmith integration for end-to-end tracing, monitoring, and debugging of the entire RAG pipeline

## 📊 Model Evaluation & Results

The system's effectiveness is validated through comprehensive evaluation using the **Ragas** framework, comparing the fine-tuned Phi-3 model against the base model on a curated financial Q&A dataset developed specifically for this domain.

### Evaluation Summary

| Metric                | Fine-Tuned Model | Base Model | Improvement (%) |
| -------------------- | ---------------- | ---------- | --------------- |
| **faithfulness**     | 0.94583          | 0.80715    | +17.18%         |
| **answer_correctness** | 0.54475        | 0.59113    | -7.85%          |
| **answer_relevancy**  | 0.96889          | 0.96889    | 0.00%           |
| **context_precision** | 1.00000          | 0.91667    | +9.09%          |
| **context_recall**    | 0.44444          | 0.44444    | 0.00%           |

The fine-tuned model demonstrates significant improvements in factual consistency (faithfulness) and context precision, crucial metrics for financial analysis applications where accuracy and reliability are paramount.

## 🏗️ Architecture

### Microservices Backend Architecture

The application implements a sophisticated microservices architecture using FastAPI with clear separation of concerns and domain-driven design principles:

#### Core Services
- **Authentication Service**: JWT-based user management with bcrypt password hashing, token generation/validation, and secure session handling
- **Conversation Service**: Multi-user conversation management with automatic naming, CRUD operations, and conversation history management
- **Assistant Service**: Core RAG pipeline orchestration using LangGraph with async/sync hybrid processing for optimal performance
- **Model Service**: Thread-safe ML model management with GPU resource locking, model loading optimization, and inference caching
- **Retrieval Service**: Hybrid search implementation combining semantic and keyword search with document reranking
- **Cache Service**: Redis-based semantic caching with configurable similarity thresholds and TTL management

#### Database Architecture
- **PostgreSQL**: Primary database for user data, conversations, and LangGraph checkpoints
- **PgBouncer**: Connection pooling layer with session-level pooling for optimal database resource utilization
- **Redis**: High-performance caching layer for semantic similarity search and session management
- **Alembic**: Database migration management for schema versioning and deployment

### The Hybrid LLM Strategy

This project employs a sophisticated dual-LLM approach, optimizing each model for specific cognitive tasks:

1. **Google Gemini Pro (Query Analysis & Reasoning)**:
   - Complex user intent analysis and query decomposition
   - Conversation context resolution and follow-up question interpretation
   - Structured metadata extraction for search filtering and optimization
   - Pydantic-based structured output generation for downstream processing
   - Temperature-controlled generation for deterministic structured outputs

2. **Fine-Tuned Phi-3 (Domain-Specific Generation)**:
   - Context-grounded financial response generation with expert domain knowledge
   - Consistent financial terminology and professional tone maintenance
   - Efficient local inference with 4-bit quantization and Unsloth optimization
   - Privacy-preserving processing without external API dependencies for sensitive financial data
   - Custom chat template optimization for financial analysis tasks

### LangGraph Workflow Architecture

![Financial Analyst Assistant Graph](Financial%20Analyst%20Assistant%20Graph.png)

The system follows an optimized LangGraph workflow designed for maximum efficiency and reliability:

1. **Query Construction Node**: Gemini-powered analysis of user intent with conversation context integration
2. **Semantic Cache Check**: Redis similarity search with configurable threshold-based cache hits
3. **Hybrid Retrieval Node**: Parallel semantic (ChromaDB) and keyword (BM25) search execution
4. **Document Reranking**: CrossEncoder-based relevance scoring and top-k selection
5. **Answer Generation Node**: Fine-tuned Phi-3 inference with context grounding
6. **Response Caching**: Automatic storage of generated responses for future similarity matching

### Production Infrastructure Patterns

- **Async/Sync Hybrid Architecture**: FastAPI async endpoints with `asyncio.to_thread()` for synchronous ML operations
- **Connection Management**: PgBouncer pooling with configurable pool sizes and connection limits
- **Resource Isolation**: Thread-safe GPU resource management with explicit locking mechanisms
- **Error Handling**: Comprehensive exception handling with graceful degradation and user-friendly error messages
- **Configuration Management**: Pydantic Settings for type-safe, validated environment configuration

## 🛠️ Technical Stack

* **FastAPI**: Serves as the high-concurrency async API framework.
* **PostgreSQL & PgBouncer**: Used for the primary database with efficient connection pooling.
* **Redis**: Provides high-performance semantic caching with vector search.
* **Docker & Docker Compose**: Manages containerization and orchestration of the entire stack.
* **Alembic**: Handles all database schema migrations.
* **Fine-tuned Phi-3**: A local model for generating domain-specific financial answers.
* **Google Gemini 1.5 Pro**: A cloud model for advanced query analysis and reasoning.
* **Unsloth**: Optimizes the local LLM for faster inference with 4-bit quantization.
* **PyTorch & CUDA**: The core machine learning framework with GPU acceleration.
* **ChromaDB & BM25**: Combined for a hybrid retrieval system (semantic + keyword search).
* **Cross-Encoder**: Reranks search results for improved relevance.
* **LangGraph**: Orchestrates the complex, multi-step RAG agent workflow.
* **LangSmith**: Provides end-to-end observability and tracing for the AI pipeline.
* **JWT**: Secures the API with token-based authentication.
* **Pydantic Settings**: Manages application configuration in a type-safe way.
* **Ragas**: Used as the framework for evaluating the RAG pipeline's performance.

**Production & Monitoring**: LangGraph workflow orchestration, LangSmith observability, JWT authentication, Pydantic Settings, Ragas evaluation framework

## 📋 Prerequisites

**Hardware**: NVIDIA GPU (CUDA 12.1+, 6GB+ VRAM), 16GB RAM, 10GB+ storage

**Software**: Python 3.11+, Docker & Docker Compose, Git, CUDA Toolkit 12.1+

**API Keys**: Google AI Studio (Gemini Pro), LangSmith monitoring, SEC EDGAR API (notebooks), OpenRouter (evaluation), Hugging Face (model access)

## 🚀 Installation & Setup

### 1. Repository Setup
```bash
git clone https://github.com/eslammohamedtolba/Financial-Insight-Engine.git
cd Financial-Insight-Engine
```

### 2. Python Environment Configuration
```bash
# Create isolated virtual environment
python -m venv venv

# Activate environment
# Windows:
.\venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install all dependencies including CUDA PyTorch
pip install -r requirements.txt
```

### 3. Environment Variables Configuration

This project uses a `.env` file to manage secrets and configurations.

1.  **Create your environment file** by copying the provided template. In your terminal, run the command that corresponds to your operating system from the block below.

    ```bash
    # For Windows (Command Prompt)
    copy .env.example .env

    # For Windows (PowerShell)
    Copy-Item .env.example .env

    # For macOS / Linux
    cp .env.example .env
    ```

2.  **Add your credentials**. Open the new `.env` file and replace the placeholder values (e.g., `your_google_api_key_here`) with your actual API keys and secrets.


### 4. Data Pipeline & Model Preparation

Execute the data preparation and model training pipeline in sequence:

1. **Knowledge Base Construction:**
   ```bash
   jupyter notebook Data/Knowledge_Base_Construction.ipynb
   ```
   - Downloads and processes SEC 10-K filings
   - Creates ChromaDB vector store with optimized chunking
   - Builds BM25 keyword search index

2. **Model Fine-Tuning:**
   ```bash
   jupyter notebook Data/Fine-Tuning_Phi-3_for_Financial_QA_with_Unsloth.ipynb
   ```
   - Fine-tunes Phi-3 model on financial Q&A dataset
   - Implements LoRA adapters for efficient training
   - Optimizes model for 4-bit inference

3. **RAG Pipeline Evaluation:**
   ```bash
   jupyter notebook Data/RAG_Pipeline_Evaluation.ipynb
   ```
   - Comprehensive evaluation using Ragas library
   - Compares fine-tuned vs base model performance
   - Generates detailed performance metrics

### 5. Docker Infrastructure Deployment

The entire production infrastructure (PostgreSQL, Redis, PgBouncer, and FastAPI application) is orchestrated using Docker Compose for simplified deployment and management.

**Build and start all services:**
```bash
# Build the application image and start all services
docker-compose up -d --build
```

**Verify service health:**
```bash
# Check that all services are running and healthy
docker-compose ps
```

Expected output should show all services as "Up" with healthy status for postgres and redis.

**View application logs:**
```bash
# Monitor real-time logs from all services
docker-compose logs -f

# View logs from specific service
docker-compose logs -f app
```

### 6. Database Schema Initialization

Initialize the production database schema using Alembic migrations:

```bash
# Run migrations inside the application container
docker-compose exec app alembic upgrade head
```

This creates all necessary tables for users, conversations, and LangGraph checkpoints.

### 7. Access the Application

Once all services are running and healthy:

- **Web Interface**: `http://localhost:8000`
- **API Documentation**: `http://localhost:8000/docs` (Swagger UI)
- **Health Check**: `http://localhost:8000/health`

**Service Ports:**
- FastAPI Application: `8000`
- PostgreSQL: `5432`
- PgBouncer: `6432`
- Redis: `6379`

### 8. Managing the Application

**Stop all services:**
```bash
docker-compose down
```

**Stop and remove all data (including volumes):**
```bash
docker-compose down -v
```

**Restart specific service:**
```bash
docker-compose restart app
```

**View resource usage:**
```bash
docker stats
```

## 📁 Project Structure

```
├── app/                                  # FastAPI application source
│   ├── assistant/                        # RAG pipeline and AI services
│   ├── authentication/                   # JWT authentication system
│   ├── conversation/                     # Multi-user conversation management
│   ├── core/                             # Shared models and schemas
│   ├── db/                               # Database configuration and sessions
│   ├── helpers/                          # Utility functions and settings
│   └── main.py                           # FastAPI application entry point
├── web-ui/                               # Frontend web interface
│   ├── css/                              # CSS styling
│   ├── js/                               # JavaScript application logic
│   ├── static/                           # Static assets
│   └── index.html                        # Main HTML entry point
├── Data/                                 # Data processing and model storage
│   ├── chroma_db/                        # (Generated) ChromaDB vector database
│   ├── phi3_finetuned_model/             # (Generated) Fine-tuned Phi-3 model
│   ├── bm25_retriever.pkl                # (Generated) BM25 keyword search index
│   ├── Knowledge_Base_Construction.ipynb                     # Data processing pipeline
│   ├── Fine-Tuning_Phi-3_for_Financial_QA_with_Unsloth.ipynb # Model finetuning pipeline
│   └── RAG_Pipeline_Evaluation.ipynb                         # Model evaluation
├── migrations/                           # Alembic database migrations
├── docker-compose.yml                    # Docker Compose orchestration
├── Dockerfile                            # Application container definition
├── .dockerignore                         # Docker build exclusions
├── requirements.txt                      # Python dependencies with CUDA PyTorch
├── alembic.ini                           # Database migration configuration
├── .env                                  # Environment variables configuration
├── .env.example                          # Environment template
└── README.md                             # Project documentation
```

## 🔒 Security & Compliance

- **Authentication**: Production-grade JWT implementation with secure token handling
- **Authorization**: User-based conversation access control with ownership verification
- **Data Privacy**: Local model inference ensures sensitive financial data never leaves your infrastructure
- **API Security**: Comprehensive CORS configuration, rate limiting, and input validation
- **Secret Management**: Environment-based configuration with Pydantic validation
- **Database Security**: Connection pooling with authentication and encrypted connections

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/advanced-rag-enhancement`
3. Commit your changes: `git commit -m 'Add advanced RAG enhancement'`
4. Push to the branch: `git push origin feature/advanced-rag-enhancement`
5. Open a Pull Request