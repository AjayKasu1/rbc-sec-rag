import os
import sys

def verify():
    print("Verifying Deployment...")
    
    # 1. Check ChromaDB
    if os.path.isdir("chroma_db") and len(os.listdir("chroma_db")) > 0:
        print("[PASS] ChromaDB directory exists and is not empty.")
    else:
        print("[FAIL] ChromaDB directory missing or empty.")
        # Don't exit yet, might be running
    
    # 2. Check Files
    required_files = ["rag_service.py", "agent_controller.py", "requirements.txt", "run.sh", "monitoring.py"]
    missing = [f for f in required_files if not os.path.exists(f)]
    if not missing:
        print("[PASS] All required files present.")
    else:
        print(f"[FAIL] Missing files: {missing}")
        
    # 3. Try to load Agent (Integration Test)
    try:
        from agent_controller import AgentController
        # Only try if DB exists
        if os.path.isdir("chroma_db"):
            print("Loading AgentController...")
            agent = AgentController()
            print("[PASS] AgentController initialized.")
            
            # Simple retrieval test
            print("Testing Retrieval...")
            docs = agent.retrieve_context("AAPL revenue", k=1)
            if docs:
                print(f"[PASS] Retrieval returned content: {docs[:50]}...")
            else:
                print("[WARN] Retrieval returned empty results (might be expected if data subset is small).")
        else:
            print("[SKIP] Agent test skipped due to missing DB.")
            
    except ImportError as e:
        print(f"[FAIL] Check dependencies: {e}")
    except Exception as e:
        print(f"[FAIL] Agent Verification Error: {e}")

if __name__ == "__main__":
    verify()
