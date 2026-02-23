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
import stat
import gc
import time

# ==========================================
# CONFIGURATION
# ==========================================
ARTIFACTS_DIR = "deploy_artifacts"
TEMP_CSV_PATH = "temp_upload.csv"

st.set_page_config(page_title="🛠️ Data Preprocessor", layout="centered")

# ==========================================
# WINDOWS FILE SYSTEM HELPERS
# ==========================================
def remove_readonly(func, path, excinfo):
    """
    Helper to force delete read-only files on Windows.
    Used by shutil.rmtree's onerror handler.
    """
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except Exception:
        pass

def robust_cleanup(dir_path):
    """
    Aggressively tries to clean up the directory, handling Windows file locks.
    """
    # 1. Force Garbage Collection to release file handles
    gc.collect()
    
    if os.path.exists(dir_path):
        # 2. Try standard deletion with permission fix
        try:
            shutil.rmtree(dir_path, onerror=remove_readonly)
        except PermissionError:
            # 3. If failed, wait 1 second and try again (Windows lag)
            time.sleep(1.0)
            try:
                shutil.rmtree(dir_path, onerror=remove_readonly)
            except Exception as e:
                st.error(f"⚠️ Could not delete existing artifacts. Please close any other apps using '{dir_path}' and try again.")
                raise e

# ==========================================
# UI & HELPER TEXT
# ==========================================
st.title("🛠️ Local Data Preprocessor")
st.markdown("""
### Instructions
1. **Close `app.py`** if it is currently running (it locks the database files).
2. Upload your CSV below.
3. This tool will overwrite `deploy_artifacts` with fresh data.
""")

# ==========================================
# PROCESSING LOGIC
# ==========================================
def process_data(uploaded_file):
    
    status_container = st.status("🚀 Processing started...", expanded=True)

    try:
        # 1. CLEANUP OLD ARTIFACTS
        status_container.write("🧹 Cleaning up old artifacts...")
        robust_cleanup(ARTIFACTS_DIR)
        os.makedirs(ARTIFACTS_DIR, exist_ok=True)

        # 2. SAVE TEMP FILE
        status_container.write("💾 Saving temporary file to disk...")
        with open(TEMP_CSV_PATH, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Connect to DuckDB
        conn = duckdb.connect()

        # 3. CONVERT TO PARQUET (With Encoding Fallback)
        status_container.write("📦 Converting CSV to optimized Parquet format...")
        parquet_path = os.path.join(ARTIFACTS_DIR, "data.parquet")
        
        try:
            conn.execute(f"""
                COPY (SELECT * FROM read_csv_auto('{TEMP_CSV_PATH}', sample_size=20000)) 
                TO '{parquet_path}' (FORMAT 'PARQUET', CODEC 'ZSTD')
            """)
        except Exception as e:
            if "Invalid Input Error" in str(e) or "unicode" in str(e).lower():
                status_container.write("⚠️ UTF-8 failed. Retrying with Latin-1 encoding...")
                conn.execute(f"""
                    COPY (SELECT * FROM read_csv_auto('{TEMP_CSV_PATH}', sample_size=20000, encoding='latin-1', ignore_errors=true)) 
                    TO '{parquet_path}' (FORMAT 'PARQUET', CODEC 'ZSTD')
                """)
            else:
                raise e

        # 4. EXTRACT METADATA
        status_container.write("🔍 Extracting schema and statistics...")
        total_rows = conn.execute(f"SELECT COUNT(*) FROM '{parquet_path}'").fetchone()[0]
        schema_df = conn.execute(f"DESCRIBE SELECT * FROM '{parquet_path}'").df()
        
        columns_meta = []
        progress_bar = status_container.progress(0)
        total_cols = len(schema_df)

        for idx, row in schema_df.iterrows():
            col_name = row['column_name']
            col_type = row['column_type']
            
            # Explicit cast to VARCHAR to handle mixed types safely
            sample_vals = conn.execute(f"""
                SELECT "{col_name}"::VARCHAR FROM '{parquet_path}' 
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

        # Save Metadata
        metadata = {"total_rows": total_rows, "columns": columns_meta}
        with open(os.path.join(ARTIFACTS_DIR, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        # 5. BUILD VECTOR STORE
        status_container.write("🧠 Building local Vector Store (Embeddings)...")
        chroma_path = os.path.join(ARTIFACTS_DIR, "chroma_db")
        
        # Initialize Client
        client = chromadb.PersistentClient(path=chroma_path)
        collection = client.create_collection("dataset_schema")
        
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

        # 6. CLEANUP & CLOSE
        # Crucial for Windows: Close DuckDB and remove temp file
        conn.close()
        
        # Force Client to release handles (best effort for Chroma)
        del client 
        del collection
        gc.collect()

        if os.path.exists(TEMP_CSV_PATH):
            try: os.remove(TEMP_CSV_PATH)
            except: pass # Non-critical if temp file stays
            
        status_container.update(label="✅ Processing Complete!", state="complete", expanded=False)
        
        st.success(f"Success! Artifacts saved to `/{ARTIFACTS_DIR}`")
        st.info("👉 You can now run `streamlit run app.py`")

    except Exception as e:
        status_container.update(label="❌ Error", state="error")
        st.error(f"An error occurred: {str(e)}")
        # Clean up connection on error
        try: conn.close()
        except: pass

# ==========================================
# MAIN INTERFACE
# ==========================================
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    st.write(f"**Filename:** {uploaded_file.name}")
    st.write(f"**Size:** {uploaded_file.size / (1024*1024):.2f} MB")
    
    if st.button("🚀 Process & Generate Artifacts"):
        process_data(uploaded_file)
