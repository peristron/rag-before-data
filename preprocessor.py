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
# FILE SYSTEM HELPERS
# ==========================================
def remove_readonly(func, path, excinfo):
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except Exception:
        pass

def robust_cleanup(dir_path):
    gc.collect() # Release file handles
    if os.path.exists(dir_path):
        try:
            shutil.rmtree(dir_path, onerror=remove_readonly)
        except PermissionError:
            time.sleep(1.0)
            try:
                shutil.rmtree(dir_path, onerror=remove_readonly)
            except Exception as e:
                st.error(f"⚠️ Locked file error. Please close any open apps using '{dir_path}'")
                raise e

# ==========================================
# UI
# ==========================================
st.title("🛠️ Local Data Preprocessor")
st.markdown("Upload your CSV. We will brute-force the encoding to make it work.")

# ==========================================
# PROCESSING LOGIC
# ==========================================
def process_data(uploaded_file):
    
    status_container = st.status("🚀 Processing started...", expanded=True)

    try:
        # 1. SETUP
        status_container.write("🧹 Cleaning up old artifacts...")
        robust_cleanup(ARTIFACTS_DIR)
        os.makedirs(ARTIFACTS_DIR, exist_ok=True)

        status_container.write("💾 Saving temporary file...")
        with open(TEMP_CSV_PATH, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        conn = duckdb.connect()
        parquet_path = os.path.join(ARTIFACTS_DIR, "data.parquet")

        # 2. CONVERT TO PARQUET (BRUTE FORCE ENCODING STRATEGY)
        status_container.write("📦 Converting CSV to Parquet...")
        
        conversion_success = False
        last_error = ""

        # List of strategies to try in order
        strategies = [
            ("UTF-8 (Auto)", f"COPY (SELECT * FROM read_csv_auto('{TEMP_CSV_PATH}', sample_size=20000)) TO '{parquet_path}' (FORMAT 'PARQUET', CODEC 'ZSTD')"),
            ("Latin-1", f"COPY (SELECT * FROM read_csv_auto('{TEMP_CSV_PATH}', sample_size=20000, encoding='latin-1')) TO '{parquet_path}' (FORMAT 'PARQUET', CODEC 'ZSTD')"),
            ("UTF-16", f"COPY (SELECT * FROM read_csv_auto('{TEMP_CSV_PATH}', sample_size=20000, encoding='utf-16')) TO '{parquet_path}' (FORMAT 'PARQUET', CODEC 'ZSTD')"),
            ("Nuclear Option (Ignore Errors)", f"COPY (SELECT * FROM read_csv_auto('{TEMP_CSV_PATH}', sample_size=20000, encoding='latin-1', ignore_errors=true)) TO '{parquet_path}' (FORMAT 'PARQUET', CODEC 'ZSTD')")
        ]

        for name, query in strategies:
            try:
                # Check if parquet already exists from previous loop and remove it
                if os.path.exists(parquet_path):
                    os.remove(parquet_path)
                
                conn.execute(query)
                status_container.write(f"✅ Success using **{name}** encoding!")
                conversion_success = True
                break # Stop trying if it works
            except Exception as e:
                last_error = str(e)
                status_container.write(f"⚠️ {name} failed, trying next strategy...")
                continue

        if not conversion_success:
            raise Exception(f"All encoding strategies failed. Last error: {last_error}")

        # 3. EXTRACT METADATA
        status_container.write("🔍 Extracting schema...")
        total_rows = conn.execute(f"SELECT COUNT(*) FROM '{parquet_path}'").fetchone()[0]
        schema_df = conn.execute(f"DESCRIBE SELECT * FROM '{parquet_path}'").df()
        
        columns_meta = []
        progress_bar = status_container.progress(0)
        total_cols = len(schema_df)

        for idx, row in schema_df.iterrows():
            col_name = row['column_name']
            col_type = row['column_type']
            
            # Safe sampling
            try:
                sample_vals = conn.execute(f"""
                    SELECT "{col_name}"::VARCHAR FROM '{parquet_path}' 
                    WHERE "{col_name}" IS NOT NULL LIMIT 3
                """).fetchall()
                samples = [str(x[0]) for x in sample_vals]
            except:
                samples = ["Could not sample"]

            desc = (
                f"Column Name: {col_name}\n"
                f"Data Type: {col_type}\n"
                f"Sample Values: {', '.join(samples)}"
            )
            
            columns_meta.append({"name": col_name, "type": col_type, "description": desc})
            progress_bar.progress((idx + 1) / total_cols)

        # Save Metadata
        metadata = {"total_rows": total_rows, "columns": columns_meta}
        with open(os.path.join(ARTIFACTS_DIR, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        # 4. VECTOR STORE
        status_container.write("🧠 Building embeddings...")
        chroma_path = os.path.join(ARTIFACTS_DIR, "chroma_db")
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

        # 5. CLEANUP
        conn.close()
        del client
        del collection
        gc.collect()
        
        status_container.update(label="✅ Processing Complete!", state="complete", expanded=False)
        st.success(f"Artifacts ready in `/{ARTIFACTS_DIR}`")
        st.info("Run `streamlit run app.py` now.")

    except Exception as e:
        status_container.update(label="❌ Critical Error", state="error")
        st.error(f"Details: {str(e)}")
        try: conn.close()
        except: pass

# ==========================================
# MAIN
# ==========================================
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
if uploaded_file and st.button("🚀 Process"):
    process_data(uploaded_file)
