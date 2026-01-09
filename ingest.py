import pandas as pd
import chromadb
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
import os
import shutil

# Configuration
DATA_PATH = "Dataset/companyfacts.csv"
CHROMA_PATH = "./chroma_db"
TARGET_CIKS = [320193, 789019]  # AAPL (320193), MSFT (789019)
TARGET_FORMS = ["10-K", "10-Q"]
MIN_YEAR = 2020

def main():
    print(f"Loading data from {DATA_PATH} in chunks...")
    
    if not os.path.exists(DATA_PATH):
        print(f"Error: {DATA_PATH} not found.")
        return

    # Use chunking to handle large CSV
    chunk_size = 100000
    filtered_chunks = []
    
    try:
        # First pass: Filter rows
        reader = pd.read_csv(
            DATA_PATH, 
            usecols=['cik', 'entityName', 'companyFact', 'val', 'fy', 'fp', 'form', 'filed', 'units', 'accn'],
            chunksize=chunk_size,
            low_memory=True # optimize
        )
        
        for i, chunk in enumerate(reader):
            if i % 10 == 0:
                print(f"Processing chunk {i}...")
            
            # fast filter
            mask = (chunk['cik'].isin(TARGET_CIKS)) & \
                   (chunk['form'].isin(TARGET_FORMS))
            
            filtered = chunk[mask]
            if not filtered.empty:
                # Secondary filter on year (requires conversion)
                filtered['fy'] = pd.to_numeric(filtered['fy'], errors='coerce')
                filtered = filtered[filtered['fy'] >= MIN_YEAR]
                
                if not filtered.empty:
                    filtered_chunks.append(filtered)
            
            # Safety break for demo speed - if we have enough data (e.g. 5000 rows found), stop scanning 13GB
            # The prompt asked for subset ~500 rows.
            total_rows = sum([len(c) for c in filtered_chunks])
            if total_rows > 2000:
                print(f"Found {total_rows} rows, stopping scan for speed.")
                break
                
    except Exception as e:
        print(f"Error processing CSV: {e}")
        return

    if not filtered_chunks:
        print("No data found after filtering.")
        return

    df = pd.concat(filtered_chunks)
    print(f"Final filtered dataframe: {len(df)} rows.")

    # Aggregate by filing (accn represents a unique filing)
    print("Aggregating into documents...")
    documents = []
    
    # Group by key filing attributes
    grouped = df.groupby(['cik', 'entityName', 'accn', 'fy', 'fp', 'form', 'filed'])

    for (cik, name, accn, fy, fp, form, filed), group in grouped:
        # Create a readable text representation of the facts in this filing
        facts_text = ""
        for _, row in group.iterrows():
            curr_val = row['val']
            # Format large numbers for readability
            if isinstance(curr_val, (int, float)) and abs(curr_val) > 1000000:
                val_str = f"{curr_val:,.0f}"
            else:
                val_str = str(curr_val)
                
            facts_text += f"- {row['companyFact']}: {val_str} {row['units']}\n"
        
        content = (
            f"Company: {name} (CIK: {cik})\n"
            f"Filing: {form} for {fp} {fy}\n"
            f"Filed Date: {filed}\n"
            f"Accession Number: {accn}\n\n"
            f"Financial Facts:\n{facts_text}"
        )
        
        metadata = {
            "cik": int(cik),
            "name": name,
            "form": form,
            "fy": int(fy) if pd.notna(fy) else 0,
            "fp": fp,
            "filed": filed,
            "source": f"{name} {form} {fy} {fp}"
        }
        
        documents.append(Document(page_content=content, metadata=metadata))

    print(f"Created {len(documents)} documents.")

    # Reset DB if exists
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    # Embed and Store
    print("Embedding and storing in ChromaDB...")
    embedding_func = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embedding_func,
        persist_directory=CHROMA_PATH
    )
    
    print(f"Successfully stored {len(documents)} documents in {CHROMA_PATH}")

if __name__ == "__main__":
    main()
