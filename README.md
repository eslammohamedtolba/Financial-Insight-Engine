# 🤖 Financial Analyst Assistant

A production-ready, enterprise-grade RAG (Retrieval-Augmented Generation) system built with modern microservices architecture. This application features a locally-run, fine-tuned **Phi-3 3.8B** model for expert financial analysis, **Google's Gemini Pro** for intelligent query processing, and comprehensive observability through **LangSmith** integration.

This sophisticated system provides intelligent analysis of the latest 10-K filings from five major technology companies (Apple, Microsoft, Google, Amazon, Meta). The architecture implements a hybrid LLM strategy, combining a specialized local model for nuanced financial insights with a powerful cloud model for complex reasoning tasks, all deployed through a scalable FastAPI backend with multi-user authentication and conversation management.

## 🖥️ User Interface

The Financial Analyst Assistant includes a web interface for user interaction:

![Financial Analyst Assistant Interface](Financial%20Analyst%20Assistant%20App.png)

The interface provides secure user authentication, multi-conversation support, and real-time chat capabilities for financial analysis queries.

## ✨ Key Features

### Production Architecture
- **Microservices Design**: Decoupled FastAPI backend with domain-driven service separation
- **Multi-User Support**: Complete JWT-based authentication and authorization system
- **Horizontal Scalability**: Stateless architecture designed for cloud deployment and auto-scaling
- **Database Connection Pooling**: PgBouncer integration for efficient PostgreSQL connection management
- **Production Security**: Comprehensive security implementation with CORS, JWT tokens, and secure secret management
- **Thread-Safe Operations**: Concurrent request handling with proper GPU resource locking for ML models

### Advanced AI/ML Architecture
- **Hybrid LLM Strategy**: Leverages specialized local model (Phi-3) for financial generation and cloud model (Gemini) for complex query analysis
- **Fine-Tuned Local Model**: `microsoft/Phi-3-mini-4k-instruct` model specifically fine-tuned on financial Q&A datasets
- **Intelligent Query Construction**: Advanced query refinement using conversation context, metadata extraction, and structured output generation
- **High-Performance Inference**: Optimized with **Unsloth** for 2x faster, memory-efficient 4-bit quantized inference
- **Advanced Reranking**: CrossEncoder model integration for optimal document relevance scoring

### Enterprise Data Management
- **Hybrid Retrieval System**: Sophisticated combination of ChromaDB vector search with BM25 keyword search for comprehensive document retrieval
- **Semantic Caching**: Redis-based similarity caching with configurable similarity thresholds to reduce latency and API costs
- **Persistent Conversation Memory**: PostgreSQL-backed conversation history with LangGraph checkpointing system
- **Document Processing Pipeline**: Automated processing of SEC 10-K filings with intelligent chunking and metadata extraction
- **Multi-Tenant Data Isolation**: Complete user data separation with conversation ownership verification

### Observability & Production Monitoring
- **Full Pipeline Tracing**: LangSmith integration for end-to-end monitoring, debugging, and performance analysis of the entire RAG pipeline
- **Structured Logging**: Comprehensive error handling and structured logging throughout all services
- **Performance Metrics**: Query latency tracking, cache hit rate monitoring, and model inference timing analysis
- **Resource Management**: GPU memory optimization, connection pool monitoring, and service health checks

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

### Backend & Infrastructure
- **API Framework**: FastAPI with async/await for high-concurrency request handling
- **Database**: PostgreSQL with async drivers (asyncpg) and connection pooling (PgBouncer)
- **Caching**: Redis with vector similarity search capabilities and TTL management
- **Migration Management**: Alembic for database schema versioning and deployment
- **Process Management**: Production-ready service orchestration and deployment patterns

### AI/ML Pipeline
- **Answer Generation**: Fine-tuned `microsoft/Phi-3-mini-4k-instruct` with domain-specific training
- **Query Analysis**: Google Gemini 1.5 Pro with structured output capabilities
- **Fine-Tuning Engine**: Unsloth for efficient model optimization with 4-bit quantization
- **ML Framework**: PyTorch with CUDA acceleration and memory optimization
- **Embeddings**: `BAAI/bge-small-en-v1.5` via FastEmbed for efficient embedding generation
- **Vector Database**: ChromaDB for semantic search with metadata filtering
- **Keyword Search**: `rank_bm25` for traditional information retrieval
- **Reranking**: `cross-encoder/ms-marco-MiniLM-L-6-v2` for document relevance optimization

### Production & Monitoring
- **Workflow Orchestration**: LangGraph for complex RAG pipeline management
- **Observability**: LangSmith for comprehensive tracing, monitoring, and debugging
- **Authentication**: JWT with secure token generation and validation
- **Configuration**: Pydantic Settings for type-safe environment management
- **Evaluation**: Ragas framework for comprehensive RAG system assessment
- **Performance**: Thread-safe model management with GPU resource optimization

## 📋 Prerequisites

### Hardware Requirements
- **NVIDIA GPU**: CUDA 12.1+ compatible GPU required for local model inference
- **VRAM**: Minimum 6GB, 8GB+ recommended for optimal Phi-3 performance
- **RAM**: 16GB system memory recommended for full pipeline operation
- **Storage**: 10GB+ free space for models, vector databases, and data storage

### Software Requirements
- **Python**: 3.11+ (64-bit) with pip and venv support
- **Docker**: Latest version with Docker Compose for infrastructure services
- **Git**: For repository management and version control
- **CUDA Toolkit**: 12.1+ for GPU acceleration and PyTorch compatibility

### API Keys & Services
Required API keys for full functionality:
- **Google AI Studio**: Gemini Pro API access for query analysis
- **LangSmith**: Monitoring and observability platform access
- **SEC EDGAR API**: Financial data retrieval (data preparation notebooks)
- **OpenRouter**: Multi-model access for evaluation (notebooks only)
- **Hugging Face**: Model downloads and fine-tuning access

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

Create a `.env` file in the root directory with production-ready configuration:

```env
# Application Settings
PROJECT_NAME="Financial Analyst Assistant"
API_V1_STR="/api/v1"

# Security Configuration
JWT_SECRET="your_secure_jwt_secret_key_here"
JWT_ALGORITHM="HS256"
ACCESS_TOKEN_EXPIRE_MINUTES=60

# Google Gemini API Configuration
GOOGLE_API_KEY="your_gemini_api_key"

# LangSmith Observability Configuration
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_API_KEY="your_langsmith_api_key"
LANGSMITH_PROJECT="your_project_name"

# Database Configuration (PgBouncer endpoints)
DATABASE_URL="postgresql+asyncpg://myuser:mypassword@localhost:6432/mydatabase"
LANGGRAPH_DATABASE_URL="postgresql://myuser:mypassword@localhost:6432/mydatabase"

# Redis Cache Configuration
REDIS_URL="redis://localhost:6379"
```

### 4. Production Infrastructure Setup

Create dedicated Docker network for service communication:
```bash
docker network create financial-analyst-net
```

Deploy production-ready database infrastructure:

**PostgreSQL Database Server:**
```bash
docker run --name my-postgres --network financial-analyst-net -d -p 5432:5432 \
  -e POSTGRES_USER=myuser \
  -e POSTGRES_PASSWORD=mypassword \
  -e POSTGRES_DB=mydatabase \
  -v postgres_data:/var/lib/postgresql/data \
  postgres:16
```

**Redis Cache Server:**
```bash
docker run --name my-redis --network financial-analyst-net -d -p 6379:6379 \
  -v redis-data:/data \
  redis/redis-stack:7.2.0-v13
```

**PgBouncer Connection Pooler:**
```bash
docker run --name my-pgbouncer --network financial-analyst-net -d -p 6432:6432 \
  -e POSTGRESQL_HOST=my-postgres \
  -e POSTGRESQL_PORT=5432 \
  -e POSTGRESQL_USERNAME=myuser \
  -e POSTGRESQL_PASSWORD=mypassword \
  -e POSTGRESQL_DATABASE=mydatabase \
  -e PGBOUNCER_DATABASE=mydatabase \
  -e PGBOUNCER_USERNAME=myuser \
  -e PGBOUNCER_PASSWORD=mypassword \
  -e PGBOUNCER_PORT=6432 \
  -e PGBOUNCER_POOL_MODE=session \
  -e PGBOUNCER_DEFAULT_POOL_SIZE=20 \
  -e PGBOUNCER_MAX_CLIENT_CONN=1000 \
  -e PGBOUNCER_MAX_DB_CONN=100 \
  -e PGBOUNCER_AUTH_TYPE=md5 \
  bitnami/pgbouncer:1.21.0
```

### 5. Data Pipeline & Model Preparation

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

### 6. Database Schema Initialization

Initialize production database schema:
```bash
# Run database migrations
alembic upgrade head
```

### 7. Production Application Launch

Start the production-ready FastAPI server:
```bash
# Development mode with hot reload
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Access the application at `http://localhost:8000`

## 📁 Project Structure

```
├── app/                                  # FastAPI application source
│   ├── __init__.py                       # Main package initializer, combines routers
│   ├── assistant/                        # RAG pipeline and AI services
│   │   ├── __init__.py                   # Makes 'assistant' a package
│   │   ├── controller/                   # Business logic and service orchestration
│   │   │   ├── __init__.py               # Makes 'controller' a package
│   │   │   ├── agent_controller.py       # Main RAG pipeline orchestration
│   │   │   ├── agent_service.py          # LangGraph workflow implementation
│   │   │   ├── base_service.py           # Abstract base class for services
│   │   │   ├── model_service.py          # Thread-safe ML model management
│   │   │   ├── retrieval_service.py      # Hybrid search implementation
│   │   │   ├── cache_service.py          # Redis semantic caching
│   │   │   └── state_service.py          # LangGraph state management
│   │   └── routes.py                     # RAG API endpoints
│   ├── authentication/                   # JWT authentication system
│   │   ├── __init__.py                   # Makes 'authentication' a package
│   │   ├── controller/                   # Auth business logic
│   │   │   ├── __init__.py               # Makes 'controller' a package
│   │   │   ├── BaseController.py         # Base auth logic (hashing, tokens)
│   │   │   └── UserController.py         # User database operations
│   │   ├── dependencies.py               # JWT middleware and validation
│   │   └── routes.py                     # Authentication API endpoints
│   ├── conversation/                     # Multi-user conversation management
│   │   ├── __init__.py                   # Makes 'conversation' a package
│   │   ├── controller/                   # Conversation business logic
│   │   │   ├── __init__.py               # Makes 'controller' a package
│   │   │   └── ConversationController.py # Conversation DB operations
│   │   └── routes.py                     # Conversation CRUD API
│   ├── core/                             # Shared models and schemas
│   │   ├── models/                     # SQLModel database models
│   │   │   ├── __init__.py             # Exposes models
│   │   │   ├── Conversation.py         # Conversation table model
│   │   │   └── User.py                 # User table model
│   │   └── schemas/                    # Pydantic API schemas
│   │       ├── __init__.py             # Exposes schemas
│   │       ├── Authentication.py       # Token-related schemas
│   │       ├── Conversation.py         # Conversation API schemas
│   │       └── User.py                 # User API schemas
│   ├── db/                               # Database configuration and sessions
│   │   ├── __init__.py                   # Makes 'db' a package
│   │   └── session.py                    # DB session dependency
│   ├── helpers/                          # Utility functions and settings
│   │   ├── __init__.py                   # Makes 'helpers' a package
│   │   └── settings.py                   # Pydantic settings management
│   └── main.py                           # FastAPI application entry point
│
├── web-ui/                             # Frontend web interface
│   ├── css/                            # CSS styling
│   │   └── styles.css                  # Main stylesheet
│   ├── js/                             # JavaScript application logic
│   │   └── app.js                      # Single-page application script
│   ├── static/                         # Static assets
│   │   └── financial-analysis.ico      # Favicon
│   └── index.html                      # Main HTML entry point
│
├── Data/                                 # Data processing and model storage
│   ├── chroma_db/                        # (Generated) ChromaDB vector database
│   ├── phi3_finetuned_model/             # (Generated) Fine-tuned Phi-3 model
│   ├── bm25_retriever.pkl                # (Generated) BM25 keyword search index
│   ├── Knowledge_Base_Construction.ipynb                     # Data processing pipeline
│   ├── Fine-Tuning_Phi-3_for_Financial_QA_with_Unsloth.ipynb # Model finetuning pipeline
│   └── RAG_Pipeline_Evaluation.ipynb                         # Model evaluation
│
├── migrations/                           # Alembic database migrations
├── requirements.txt                      # Python dependencies with CUDA PyTorch
├── alembic.ini                           # Database migration configuration
├── .env                                  # Environment variables configuration
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
