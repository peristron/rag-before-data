# ==========================================
# 🛠️ PREPROCESSOR (LOCAL INGESTION ENGINE) - V16 (chattier UI)
# ==========================================
# UPDATES:
#   1. UX: Added granular status updates during Unzip/Copy phases.
#   2. UX: Added "Patience" warning for large files.
#   3. Logic: Retains Hybrid Mode (Merge/Multi) and Auto-Git.
# ==========================================

import streamlit as st
import duckdb
import pandas as pd
import json
import os
import shutil
import chromadb
from sentence_transformers import SentenceTransformer
import stat
import gc
import time
import math
import zipfile
import subprocess
import re

# ==========================================
# CONFIGURATION
# ==========================================
DEFAULT_ARTIFACTS_DIR = "deploy_artifacts"
TEMP_DIR = "temp_uploads"
TEMP_MASTER_PARQUET = "temp_master.parquet"
CHUNK_SIZE_MB = 90

st.set_page_config(page_title="🛠️ RAG Hybrid Engine V16", layout="centered")

# ==========================================
# UI HEADER
# ==========================================
st.title("🛠️ RAG Ingestion Engine")
st.caption("Local ETL Pipeline for Big Data RAG")

with st.expander("ℹ️ **How to use this tool**", expanded=False):
    st.markdown("""
    ### 1. Select Strategy
    *   **Merge All:** Best for time-series splits (e.g., `Jan.csv` + `Feb.csv`).
    *   **Keep Separate:** Best for relational data (e.g., `Customers.csv` + `Orders.csv`).

    ### 2. Choose Input
    *   **Browser:** Good for small files (<200MB).
    *   **Local Path:** Mandatory for huge files (>1GB).

    ### 3. Deploy
    *   Check **Auto-Deploy** to push directly to GitHub.
    """)

# ==========================================
# SYSTEM HELPERS
# ==========================================
def remove_readonly(func, path, excinfo):
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except Exception: pass

def robust_cleanup(dir_path):
    gc.collect()
    if os.path.exists(dir_path):
        try: shutil.rmtree(dir_path, onerror=remove_readonly)
        except:
            time.sleep(1.0)
            try: shutil.rmtree(dir_path, onerror=remove_readonly)
            except: pass
    if os.path.exists(TEMP_DIR):
        try: shutil.rmtree(TEMP_DIR, onerror=remove_readonly)
        except: pass
    if os.path.exists(TEMP_MASTER_PARQUET):
        try: os.remove(TEMP_MASTER_PARQUET)
        except: pass

def sanitize_table_name(filename):
    name = os.path.splitext(filename)[0]
    clean = re.sub(r'[^a-zA-Z0-9]', '_', name).lower()
    return clean.strip('_')

def get_all_csvs(root_dir):
    csv_files = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(".csv"):
                csv_files.append(os.path.join(root, file))
    return csv_files

def run_git_sync():
    st.write("🐙 **Starting Git Sync...**")
    terminal = st.empty()
    commands = [
        ["git", "add", "."],
        ["git", "commit", "-m", "Updated data artifacts via Preprocessor"],
        ["git", "push"]
    ]
    try:
        for cmd in commands:
            cmd_str = " ".join(cmd)
            terminal.code(f"> {cmd_str}")
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
            if result.returncode != 0:
                if "nothing to commit" in result.stdout or "nothing to commit" in result.stderr:
                    st.info("Nothing to change in Git (files are identical).")
                else:
                    raise Exception(f"Git Error: {result.stderr}")
            else:
                st.success(f"Executed: {cmd_str}")
        terminal.empty()
        return True
    except Exception as e:
        st.error(f"Git Automation Failed: {e}")
        return False

# ==========================================
# LOGIC A: MERGE ALL
# ==========================================
def process_merge_strategy(conn, all_csvs, status):
    status.write("🔗 **Strategy: Merge All**...")
    
    input_files_sql = [f"'{f.replace(os.sep, '/')}'" for f in all_csvs]
    input_pattern = ", ".join(input_files_sql)
    
    conversion_success = False
    strategies = [
        ("UTF-8", f"COPY (SELECT * FROM read_csv_auto([{input_pattern}], sample_size=100000)) TO '{TEMP_MASTER_PARQUET}' (FORMAT 'PARQUET', CODEC 'ZSTD')"),
        ("Latin-1", f"COPY (SELECT * FROM read_csv_auto([{input_pattern}], sample_size=100000, encoding='latin-1')) TO '{TEMP_MASTER_PARQUET}' (FORMAT 'PARQUET', CODEC 'ZSTD')"),
        ("Ignore Errors", f"COPY (SELECT * FROM read_csv_auto([{input_pattern}], sample_size=100000, encoding='latin-1', ignore_errors=true)) TO '{TEMP_MASTER_PARQUET}' (FORMAT 'PARQUET', CODEC 'ZSTD')")
    ]
    
    for name, query in strategies:
        try:
            if os.path.exists(TEMP_MASTER_PARQUET): os.remove(TEMP_MASTER_PARQUET)
            conn.execute(query)
            status.write(f"✅ Success using **{name}** strategy!")
            conversion_success = True
            break
        except: continue

    if not conversion_success:
        status.write("⚠️ SQL engines failed. Using Pandas Sanitization...")
        try:
            chunk_size = 200000
            first_chunk = True
            for csv_file in all_csvs:
                with pd.read_csv(csv_file, chunksize=chunk_size, encoding_errors='replace', on_bad_lines='skip') as reader:
                    for chunk in reader:
                        chunk.columns = chunk.columns.astype(str).str.strip().str.replace('"', '')
                        if first_chunk:
                            chunk.to_parquet(TEMP_MASTER_PARQUET, engine='pyarrow', index=False)
                            first_chunk = False
                        else:
                            chunk.to_parquet(TEMP_MASTER_PARQUET, engine='pyarrow', index=False, append=True)
        except Exception as e: raise Exception(f"Failed to process CSVs: {e}")

    file_size_mb = os.path.getsize(TEMP_MASTER_PARQUET) / (1024 * 1024)
    total_rows = conn.execute(f"SELECT COUNT(*) FROM '{TEMP_MASTER_PARQUET}'").fetchone()[0]
    
    table_name = "data"
    
    if file_size_mb < CHUNK_SIZE_MB:
        os.rename(TEMP_MASTER_PARQUET, os.path.join(DEFAULT_ARTIFACTS_DIR, f"{table_name}_0.parquet"))
    else:
        num_chunks = math.ceil(file_size_mb / CHUNK_SIZE_MB)
        rows_per_chunk = math.ceil(total_rows / num_chunks)
        status.write(f"✂️ Splitting into {num_chunks} parts...")
        for i in range(num_chunks):
            offset = i * rows_per_chunk
            chunk_path = os.path.join(DEFAULT_ARTIFACTS_DIR, f"{table_name}_{i}.parquet")
            conn.execute(f"COPY (SELECT * FROM '{TEMP_MASTER_PARQUET}' LIMIT {rows_per_chunk} OFFSET {offset}) TO '{chunk_path}' (FORMAT 'PARQUET', CODEC 'ZSTD')")
        if os.path.exists(TEMP_MASTER_PARQUET): os.remove(TEMP_MASTER_PARQUET)

    first_chunk = os.path.join(DEFAULT_ARTIFACTS_DIR, f"{table_name}_0.parquet")
    schema_df = conn.execute(f"DESCRIBE SELECT * FROM '{first_chunk}'").df()
    columns_meta = []
    for _, row in schema_df.iterrows():
        col, dtype = row['column_name'], row['column_type']
        try:
            samples = conn.execute(f"""SELECT "{col}"::VARCHAR FROM '{first_chunk}' WHERE "{col}" IS NOT NULL LIMIT 3""").fetchall()
            sample_str = ", ".join([str(x[0]) for x in samples])
        except: sample_str = "N/A"
        desc = f"Table: {table_name}\nColumn: {col}\nType: {dtype}\nSamples: {sample_str}"
        columns_meta.append({"name": col, "type": dtype, "description": desc, "table": table_name})

    return {"tables": {table_name: {"file_pattern": f"{table_name}_*.parquet", "total_rows": total_rows}}, "columns": columns_meta}

# ==========================================
# LOGIC B: MULTI-TABLE LOOP
# ==========================================
def process_multi_strategy(conn, all_csvs, status):
    status.write("🧩 **Strategy: Keep Separate Tables**...")
    
    tables_metadata = {}
    all_columns_meta = []
    prog_bar = status.progress(0)
    
    for idx, csv_file in enumerate(all_csvs):
        raw_name = os.path.basename(csv_file)
        table_name = sanitize_table_name(raw_name)
        
        status.write(f"⚙️ Processing table **{idx+1}/{len(all_csvs)}**: `{table_name}`...")
        
        temp_parquet = os.path.join(TEMP_DIR, f"{table_name}_temp.parquet")
        input_path = csv_file.replace(os.sep, '/')
        
        try:
            conn.execute(f"COPY (SELECT * FROM read_csv_auto('{input_path}', sample_size=100000)) TO '{temp_parquet}' (FORMAT 'PARQUET', CODEC 'ZSTD')")
        except:
            chunk_size = 200000
            first_chunk = True
            with pd.read_csv(csv_file, chunksize=chunk_size, encoding_errors='replace', on_bad_lines='skip') as reader:
                for chunk in reader:
                    chunk.columns = chunk.columns.astype(str).str.strip().str.replace('"', '')
                    if first_chunk:
                        chunk.to_parquet(temp_parquet, engine='pyarrow', index=False)
                        first_chunk = False
                    else:
                        chunk.to_parquet(temp_parquet, engine='pyarrow', index=False, append=True)

        file_size_mb = os.path.getsize(temp_parquet) / (1024 * 1024)
        total_rows = conn.execute(f"SELECT COUNT(*) FROM '{temp_parquet}'").fetchone()[0]
        
        if file_size_mb < CHUNK_SIZE_MB:
            os.rename(temp_parquet, os.path.join(DEFAULT_ARTIFACTS_DIR, f"{table_name}_0.parquet"))
        else:
            num_chunks = math.ceil(file_size_mb / CHUNK_SIZE_MB)
            rows_per_chunk = math.ceil(total_rows / num_chunks)
            status.write(f"✂️ Splitting `{table_name}` into {num_chunks} parts...")
            for i in range(num_chunks):
                offset = i * rows_per_chunk
                chunk_path = os.path.join(DEFAULT_ARTIFACTS_DIR, f"{table_name}_{i}.parquet")
                conn.execute(f"COPY (SELECT * FROM '{temp_parquet}' LIMIT {rows_per_chunk} OFFSET {offset}) TO '{chunk_path}' (FORMAT 'PARQUET', CODEC 'ZSTD')")
            os.remove(temp_parquet)

        first_chunk = os.path.join(DEFAULT_ARTIFACTS_DIR, f"{table_name}_0.parquet")
        schema_df = conn.execute(f"DESCRIBE SELECT * FROM '{first_chunk}'").df()
        for _, row in schema_df.iterrows():
            col, dtype = row['column_name'], row['column_type']
            try:
                samples = conn.execute(f"""SELECT "{col}"::VARCHAR FROM '{first_chunk}' WHERE "{col}" IS NOT NULL LIMIT 3""").fetchall()
                sample_str = ", ".join([str(x[0]) for x in samples])
            except: sample_str = "N/A"
            desc = f"Table: {table_name}\nColumn: {col}\nType: {dtype}\nSamples: {sample_str}"
            all_columns_meta.append({"name": col, "type": dtype, "description": desc, "table": table_name})

        tables_metadata[table_name] = {"file_pattern": f"{table_name}_*.parquet", "total_rows": total_rows}
        prog_bar.progress((idx + 1) / len(all_csvs))
        
    return {"tables": tables_metadata, "columns": all_columns_meta}

# ==========================================
# MAIN CONTROLLER
# ==========================================
def process_data(inputs, input_type="upload", strategy="merge", auto_push=False):
    status = st.status("🚀 Processing started...", expanded=True)
    start_time = time.time()

    try:
        status.write("🧹 Cleaning up workspace...")
        robust_cleanup(DEFAULT_ARTIFACTS_DIR)
        os.makedirs(DEFAULT_ARTIFACTS_DIR, exist_ok=True)
        os.makedirs(TEMP_DIR, exist_ok=True)

        status.write(f"💾 Ingesting {len(inputs)} files/paths...")
        
        # --- GRANULAR COPY/UNZIP LOGIC ---
        for i, item in enumerate(inputs):
            fname = ""
            file_path = ""
            
            if input_type == "upload":
                fname = item.name
                status.write(f"📥 Reading upload: {fname}...")
                file_path = os.path.join(TEMP_DIR, fname)
                with open(file_path, "wb") as f: f.write(item.getbuffer())
            else:
                src_path = item.strip('"').strip("'")
                fname = os.path.basename(src_path)
                status.write(f"📥 Copying local file: {fname}...")
                
                if not os.path.exists(src_path): raise Exception(f"File not found: {src_path}")
                file_path = os.path.join(TEMP_DIR, fname)
                try: shutil.copy2(src_path, file_path)
                except Exception as e: raise Exception(f"Could not copy file: {e}")

            if fname.lower().endswith(".zip"):
                status.write(f"📂 Unzipping {fname} (This may take a moment)...")
                with zipfile.ZipFile(file_path, 'r') as zip_ref: zip_ref.extractall(TEMP_DIR)
                os.remove(file_path)
        # ---------------------------------

        all_csvs = get_all_csvs(TEMP_DIR)
        if not all_csvs: raise Exception("No CSV files found!")
        
        status.write(f"found {len(all_csvs)} CSVs to process.")
        conn = duckdb.connect()
        
        if strategy == "merge":
            result = process_merge_strategy(conn, all_csvs, status)
        else:
            result = process_multi_strategy(conn, all_csvs, status)

        with open(os.path.join(DEFAULT_ARTIFACTS_DIR, "metadata.json"), "w") as f:
            json.dump({"tables": result['tables']}, f, indent=2)

        status.write("🧠 Building embeddings...")
        chroma_path = os.path.join(DEFAULT_ARTIFACTS_DIR, "chroma_db")
        if os.path.exists(chroma_path):
            try: shutil.rmtree(chroma_path, onerror=remove_readonly)
            except: pass
        
        try:
            client = chromadb.PersistentClient(path=chroma_path)
            collection = client.create_collection("dataset_schema")
            model = SentenceTransformer('all-MiniLM-L6-v2')
            
            docs = [c["description"] for c in result['columns']]
            ids = [f"{c['table']}.{c['name']}" for c in result['columns']]
            metadatas = [{"name": c["name"], "type": c["type"], "table": c["table"]} for c in result['columns']]
            
            embs = model.encode(docs).tolist()
            collection.add(documents=docs, embeddings=embs, ids=ids, metadatas=metadatas)
        except Exception as e: raise Exception(f"ChromaDB Error: {str(e)}")

        conn.close()
        try: del client
        except: pass
        gc.collect()
        if os.path.exists(TEMP_DIR):
            try: shutil.rmtree(TEMP_DIR, onerror=remove_readonly)
            except: pass
            
        elapsed_time = time.time() - start_time
        status.update(label="✅ Processing Complete!", state="complete", expanded=False)
        st.success(f"**Success!** Artifacts saved to `{DEFAULT_ARTIFACTS_DIR}`.")
        st.info(f"⏱️ **Total Execution Time:** {elapsed_time:.2f} seconds")

        if auto_push:
            if run_git_sync():
                st.balloons()
                st.success("🚀 **Deployed!** Changes pushed to GitHub.")

    except Exception as e:
        status.update(label="❌ Critical Error", state="error")
        st.error(f"Details: {str(e)}")
        try: conn.close()
        except: pass

# ==========================================
# MAIN UI
# ==========================================
st.write("### 1. Configuration")

strategy = st.radio(
    "Processing Strategy",
    ["Merge All (Single Table)", "Keep Separate (Multi-Table)"],
    index=0,
    help="Merge All: Stitches CSVs into one dataset. Keep Separate: Creates distinct tables for JOINs."
)
strategy_key = "merge" if "Merge" in strategy else "multi"

if strategy_key == "merge":
    st.info("ℹ️ **Will happen:** All found CSVs will be stacked into **one** giant table named `data`.")
else:
    st.info("ℹ️ **Will happen:** Each CSV will become a **separate** SQL table (e.g., `customers`, `orders`) enabling relational queries.")

auto_deploy = st.checkbox("🔄 **Auto-Deploy:** Push to GitHub immediately after processing?", value=False)

st.write("### 2. Upload Data")
st.warning("⚠️ **Note:** Unzipping & Processing GB-scale files takes time. Please be patient and do not close this tab.")

tab1, tab2 = st.tabs(["📂 Browser Upload", "🛣️ Local File Path"])

with tab1:
    st.markdown("Use this for files **under 200MB**.")
    uploaded_files = st.file_uploader("Drag & Drop CSVs or ZIPs", type=["csv", "zip"], accept_multiple_files=True)
    if uploaded_files and st.button("🚀 Process Uploads"):
        process_data(uploaded_files, input_type="upload", strategy=strategy_key, auto_push=auto_deploy)

with tab2:
    st.markdown("Use this for **Huge Files (GBs)**. It skips browser loading.")
    local_path = st.text_input("Paste Full File Path", placeholder=r"C:\Users\YourName\Downloads\BigData.zip")
    if local_path and st.button("🚀 Process Local Path"):
        process_data([local_path], input_type="path", strategy=strategy_key, auto_push=auto_deploy)
