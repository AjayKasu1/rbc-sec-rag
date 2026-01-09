# Capital Markets RAG Service

A production-style GenAI RAG service for Capital Markets Research, utilizing SEC data.

## Features
- **RAG Engine**: Ingests and retrieves financial facts from 10-K/10-Q filings.
- **Agentic Controller**: Reasoning loop that can use tools (e.g., Financial Ratio Calculator).
- **Evaluation**: pipeline for evaluating answer quality.
- **Monitoring**: Dashboard for tracking queries and latency.

## Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Ingest Data**:
   Ensure `Dataset/companyfacts.csv` is present.
   ```bash
   python ingest.py
   ```

3. **Run Service**:
   ```bash
   uvicorn rag_service:app --reload
   ```

4. **Run Monitoring Dashboard**:
   ```bash
   streamlit run monitoring.py
   ```

## Key Files
- `ingest.py`: Data processing and vector store creation.
- `rag_service.py`: FastAPI backend.
- `agent_controller.py`: Core logic for RAG and tools.
- `eval_pipeline.py`: Automated evaluation script.

## Resume Points
- Implemented high-performance GenAI RAG service for SEC filings using LangChain and ChromaDB.
- Deployed agentic controller with custom tools for real-time financial ratio analysis.
- Designed comprehensive evaluation pipeline with LLM-as-a-judge metrics and monitoring dashboard.
