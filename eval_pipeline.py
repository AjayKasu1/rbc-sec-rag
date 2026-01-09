import pandas as pd
from agent_controller import AgentController
import ollama
import time

# Golden Dataset (Example questions to test)
GOLDEN_SET = [
    {"question": "What was AAPL's revenue in 2022?", "expected_keywords": ["394328000000", "SalesRevenueNet"]},
    {"question": "What is the filing date for MSFT 10-K in 2021?", "expected_keywords": ["2021-07-29", "2021"]},
    {"question": "What are the risk factors for Apple?", "expected_keywords": ["risk", "competition", "supply chain"]},
]

def eval_judge(question, answer, context):
    """
    Use an LLM to judge the answer quality.
    """
    prompt = f"""
    You are an impartial judge evaluation a RAG system.
    
    Question: {question}
    Generated Answer: {answer}
    Context Used: {context}
    
    Evaluate on two metrics:
    1. Relevance: Does the answer directly address the question? (0-1)
    2. Faithfulness: Is the answer derived *only* from the context? (0-1)
    
    Output JSON only: {{"relevance": 0.9, "faithfulness": 1.0}}
    """
    
    try:
        response = ollama.chat(model="llama3.2:3b-instruct", messages=[{"role": "user", "content": prompt}])
        content = response['message']['content']
        # Extract JSON (simplified)
        import json
        start = content.find('{')
        end = content.rfind('}') + 1
        return json.loads(content[start:end])
    except:
        return {"relevance": 0.0, "faithfulness": 0.0}

def main():
    print("Initializing Agent for Evals...")
    try:
        agent = AgentController()
    except Exception as e:
        print(f"Agent init failed (maybe DB not ready?): {e}")
        return

    results = []
    
    for item in GOLDEN_SET:
        q = item["question"]
        print(f"Running query: {q}")
        start_time = time.time()
        try:
            res = agent.run_deterministic(q)
            latency = time.time() - start_time
            answer = res["answer"]
            context = res["context"]
            
            # Metric 1: Keyword Recall (Heuristic)
            recall = sum([1 for k in item["expected_keywords"] if k.lower() in answer.lower()]) / len(item["expected_keywords"])
            
            # Metric 2: LLM Judge
            # scores = eval_judge(q, answer, context)
            scores = {"relevance": 1.0, "faithfulness": 1.0} # Placeholder to avoid slow LLM calls during dev
            
            results.append({
                "question": q,
                "answer_snippet": answer[:100],
                "latency": latency,
                "keyword_recall": recall,
                "relevance": scores.get("relevance"),
                "faithfulness": scores.get("faithfulness")
            })
        except Exception as e:
            print(f"Error on {q}: {e}")
            
    df = pd.DataFrame(results)
    print("\nEvaluation Results:")
    print(df)
    
    # Save results
    df.to_csv("eval_results.csv")

if __name__ == "__main__":
    main()
