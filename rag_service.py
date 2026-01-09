from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from agent_controller import AgentController
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Capital Markets RAG Service")

# Initialize agent (lazy load or startup)
agent = None

@app.on_event("startup")
def startup_event():
    global agent
    logger.info("Starting up RAG Service...")
    try:
        agent = AgentController()
        logger.info("Agent initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize agent: {e}")

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    context_used: str

@app.post("/answer", response_model=QueryResponse)
async def answer_question(request: QueryRequest):
    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    logger.info(f"Received question: {request.question}")
    
    try:
        # Using the deterministic run for stability in the demo
        result = agent.run_deterministic(request.question)
        return QueryResponse(
            answer=result["answer"],
            context_used=result["context"]
        )
    except Exception as e:
        logger.error(f"Error processing request: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
