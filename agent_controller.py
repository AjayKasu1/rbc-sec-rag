from typing import List, Dict, Any, Optional
import json
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import ollama

# Configuration
CHROMA_PATH = "./chroma_db"
LLM_MODEL = "llama3.2:3b"

class AgentController:
    def __init__(self):
        print("Initializing Agent Controller...")
        self.embedding_func = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.vector_store = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=self.embedding_func
        )
        self.tools = {
            "retrieve_context": self.retrieve_context,
            "calc_financial_ratio": self.calc_financial_ratio
        }
        
    def retrieve_context(self, query: str, k: int = 3) -> str:
        """Retrieve relevant context from SEC filings."""
        print(f"Tool Call: retrieve_context('{query}')")
        results = self.vector_store.similarity_search(query, k=k)
        context = ""
        for i, doc in enumerate(results):
            context += f"\n--- Document {i+1} ---\n{doc.page_content}\n"
        return context

    def calc_financial_ratio(self, numerator: float, denominator: float, ratio_name: str) -> str:
        """Calculate a financial ratio given two numbers."""
        print(f"Tool Call: calc_financial_ratio({numerator}, {denominator})")
        if denominator == 0:
            return f"Error: Cannot calculate {ratio_name} (division by zero)."
        ratio = numerator / denominator
        return f"Calculated {ratio_name}: {ratio:.4f}"

    def run(self, user_query: str) -> Dict[str, Any]:
        """Run the agent loop."""
        messages = [
            # System prompt defining the persona and tools
            {
                "role": "system", 
                "content": """You are a financial analyst assistant for valid Capital Markets Research. 
You have access to the following tools:
1. retrieve_context(query: str): Search for facts in SEC filings.
2. calc_financial_ratio(numerator: float, denominator: float, ratio_name: str): Compute a ratio.

Use the tools to answer the user's question accurately. 
Context retrieval is your primary source of truth. 
If you need to calculate a ratio, find the numbers first using retrieval, then use the calculation tool.
Always cite the filing (Form, Year) when providing facts.
"""
            },
            {"role": "user", "content": user_query}
        ]

        # Simple ReAct-style loop (simplified for demonstration with Ollama)
        # We will do a max of 3 turns
        
        for turn in range(3):
            print(f"--- Turn {turn + 1} ---")
            response = ollama.chat(model=LLM_MODEL, messages=messages)
            content = response['message']['content']
            messages.append(response['message'])
            
            # Simple heuristic to detect tool calls (since local models might not output structured JSON reliably)
            # We look for textual indicators or JSON blocks if the model is fine-tuned for it.
            # Llama 3.2 is decent. Let's try to parse tool calls if they follow a specific format or JSON.
            
            # For this quick implementation, we'll strip markers or look for specific patterns
            # Or better, we force the model to output JSON in the system prompt for better control?
            # Given "llama3.2:3b", it's safer to prompt it to output a specific JSON structure for actions.
            
            # Let's adjust the system prompt slightly in a real implementation, but for now we'll assume
            # the model might just textually describe what it wants. 
            # OR we can assume the user just wants a direct answer for simple RAG.
            
            # But the prompt requires "Controller retrieves -> optional tool -> compliance -> answer".
            # Let's check if the model asks to use a tool.
            
            # Basic Tool Parsing Logic (Naive)
            if "retrieve_context" in content and "(" in content:
                # Extract query
                try:
                    start = content.find("retrieve_context('") + 18
                    end = content.find("')", start)
                    if start == 17: # Try double quotes
                         start = content.find('retrieve_context("') + 18
                         end = content.find('")', start)
                    
                    if start > 17 and end > start:
                        query = content[start:end]
                        tool_result = self.retrieve_context(query)
                        messages.append({"role": "user", "content": f"Tool Output: {tool_result}"})
                        continue
                except:
                    pass
            
            if "calc_financial_ratio" in content:
                # Naive parsing for calc
                pass

            # If no tool called, or we decide to stop
            if "Answer:" in content or turn == 2:
                final_answer = content.split("Answer:")[-1].strip() if "Answer:" in content else content
                return {"answer": final_answer, "log": messages}
                
        return {"answer": messages[-1]['content'], "log": messages}

    # Better approach for reliable tool use with small local models:
    # 1. Force retrieval first (Classic RAG)
    # 2. Then let LLM decide if it needs calculation
    
    def run_deterministic(self, user_query: str) -> Dict[str, Any]:
        """A more robust pipeline for the demo: Retrieve -> Reason -> Answer"""
        
        # Step 1: Retrieve
        context = self.retrieve_context(user_query)
        
        # Step 2: Ask LLM if it needs to calculate anything
        # (Optimization: We can skip this if the query doesn't look like a math question, but let's be safe)
        
        prompt = f"""Context:
{context}

User Question: {user_query}

Based on the context, answer the question. 
If you need to calculate a financial ratio and have the numbers, show the calculation step.
If you don't have enough information, state what is missing.
Format your answer as:
Thought: ...
Answer: ...
"""
        response = ollama.chat(model=LLM_MODEL, messages=[{"role": "user", "content": prompt}])
        return {"answer": response['message']['content'], "context": context}

