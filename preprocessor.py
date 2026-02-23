# preprocess.py

# # requirements.txt streamlit pandas duckdb openai chromadb sentence-transformers pyarrow
# run command:
#                             streamlit run preprocess.py
#                  directory setup: cd C:\Users\oakhtar\OneDrive - D2L Corporation\Documents\pyprojs_local

import streamlit as st
import duckdb
import json
import os
import shutil
import chromadb
from sentence_transformers import SentenceTransformer

# ==========================================
# CONFIGURATION
# ==========================================
ARTIFACTS_DIR = "deploy_artifacts"
TEMP_CSV_PATH = "temp_upload.csv"

st.set_page_config(page_title="🛠️ Data Preprocessor", layout="centered")

# ==========================================
# UI & HELPER TEXT
# ==========================================
st.title("🛠️ Local Data Preprocessor")
st.markdown("""
### What is this?
This tool prepares your **large CSV files** for RAG deployment.
It performs three heavy operations locally so your cloud app doesn't have to:

1.  **Compression:** Converts CSV to **Parquet** (Reduces size by ~70%, speeds up queries 10x).
2.  **Metadata Extraction:** Scans columns to understand the schema.
3.  **Vector Indexing:** Embeds column descriptions so the AI knows which columns to query.

**Output:** A folder named `deploy_artifacts` containing optimized files.
""")

# ==========================================
# PROCESSING LOGIC
# ==========================================
def process_data(uploaded_file):
    
    # Create/Reset Artifacts Directory
    if os.path.exists(ARTIFACTS_DIR):
        shutil.rmtree(ARTIFACTS_DIR)
    os.makedirs(ARTIFACTS_DIR)

    # Status Container
    status_container = st.status("🚀 Processing started...", expanded=True)

    try:
        # 1. SAVE TEMP FILE
        # DuckDB prefers file paths over memory buffers for large files
        status_container.write("💾 Saving temporary file to disk...")
        with open(TEMP_CSV_PATH, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        conn = duckdb.connect()

        # 2. CONVERT TO PARQUET
        status_container.write("📦 Converting CSV to optimized Parquet format...")
        parquet_path = os.path.join(ARTIFACTS_DIR, "data.parquet")
        
        # We sample 20k rows to infer types, then write the whole file
        conn.execute(f"""
            COPY (SELECT * FROM read_csv_auto('{TEMP_CSV_PATH}', sample_size=20000)) 
            TO '{parquet_path}' (FORMAT 'PARQUET', CODEC 'ZSTD')
        """)

        # 3. EXTRACT METADATA
        status_container.write("🔍 Extracting schema and statistics...")
        total_rows = conn.execute(f"SELECT COUNT(*) FROM '{parquet_path}'").fetchone()[0]
        schema_df = conn.execute(f"DESCRIBE SELECT * FROM '{parquet_path}'").df()
        
        columns_meta = []
        progress_bar = status_container.progress(0)
        total_cols = len(schema_df)

        for idx, row in schema_df.iterrows():
            col_name = row['column_name']
            col_type = row['column_type']
            
            # Get sample values for context (crucial for LLM)
            sample_vals = conn.execute(f"""
                SELECT "{col_name}" FROM '{parquet_path}' 
                WHERE "{col_name}" IS NOT NULL LIMIT 3
            """).fetchall()
            samples = [str(x[0]) for x in sample_vals]
            
            desc = (
                f"Column Name: {col_name}\n"
                f"Data Type: {col_type}\n"
                f"Sample Values: {', '.join(samples)}"
            )
            
            columns_meta.append({
                "name": col_name,
                "type": col_type,
                "description": desc
            })
            progress_bar.progress((idx + 1) / total_cols)

        # Save Metadata JSON
        metadata = {"total_rows": total_rows, "columns": columns_meta}
        with open(os.path.join(ARTIFACTS_DIR, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        # 4. BUILD VECTOR STORE
        status_container.write("🧠 Building local Vector Store (Embeddings)...")
        chroma_path = os.path.join(ARTIFACTS_DIR, "chroma_db")
        client = chromadb.PersistentClient(path=chroma_path)
        collection = client.create_collection("dataset_schema")
        
        # Local Embedding Model (No API Key needed)
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        documents = [c["description"] for c in columns_meta]
        ids = [c["name"] for c in columns_meta]
        embeddings = model.encode(documents).tolist()
        
        collection.add(
            documents=documents,
            embeddings=embeddings,
            ids=ids,
            metadatas=[{"name": c["name"], "type": c["type"]} for c in columns_meta]
        )

        # Cleanup
        os.remove(TEMP_CSV_PATH)
        status_container.update(label="✅ Processing Complete!", state="complete", expanded=False)
        
        st.success(f"Success! Artifacts saved to `/{ARTIFACTS_DIR}`")
        st.info("👉 You can now run `streamlit run app.py` to chat with your data.")

    except Exception as e:
        status_container.update(label="❌ Error", state="error")
        st.error(f"An error occurred: {str(e)}")

# ==========================================
# MAIN INTERFACE
# ==========================================
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    # Display file stats
    st.write(f"**Filename:** {uploaded_file.name}")
    st.write(f"**Size:** {uploaded_file.size / (1024*1024):.2f} MB")
    
    if st.button("🚀 Process & Generate Artifacts"):
        process_data(uploaded_file)
