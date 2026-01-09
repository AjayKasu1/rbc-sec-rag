import streamlit as st
import pandas as pd
import sqlite3
import time
import os

# Configuration
EVAL_RESULTS_PATH = "eval_results.csv"

st.set_page_config(page_title="RAG Monitoring Dashboard", layout="wide")

st.title("Capital Markets RAG Service - Monitoring")

# Tabs
tab1, tab2 = st.tabs(["Real-time Query", "Evaluation Metrics"])

with tab1:
    st.header("Test the Agent")
    question = st.text_input("Ask a question about AAPL or MSFT financial facts:")
    
    if st.button("Submit"):
        if question:
            try:
                # Call the API directly or import agent (simulated for dashboard if API is running)
                import requests
                response = requests.post("http://localhost:8000/answer", json={"question": question})
                
                if response.status_code == 200:
                    data = response.json()
                    st.markdown(f"**Answer:** {data['answer']}")
                    with st.expander("View Context"):
                        st.text(data['context_used'])
                else:
                    st.error(f"API Error: {response.text}")
            except Exception as e:
                st.error(f"Failed to connect to RAG service: {e}")
                
with tab2:
    st.header("Offline Evaluation Results")
    
    if os.path.exists(EVAL_RESULTS_PATH):
        df = pd.read_csv(EVAL_RESULTS_PATH)
        
        # KPI Cards
        col1, col2, col3 = st.columns(3)
        col1.metric("Avg Latency", f"{df['latency'].mean():.2f}s")
        col2.metric("Avg Keyword Recall", f"{df['keyword_recall'].mean():.2f}")
        col3.metric("Queries Tested", len(df))
        
        st.dataframe(df)
        
        st.subheader("Latency Distribution")
        st.bar_chart(df['latency'])
    else:
        st.info("No evaluation results found. Run `eval_pipeline.py` first.")
