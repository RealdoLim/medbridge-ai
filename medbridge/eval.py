import sys
import os

# Add the main 'medbridge-ai' root folder to Python's radar
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
import time

# Import RAG function
from rag import answer_query 

st.set_page_config(page_title="MedBridge Eval", layout="wide")
st.title("Mission 9: RAG Evaluation Dashboard")

# Test questions for evaluation. Each question has an 'expected_keyword' that should 
# appear in the retrieved documents for it to be considered a "Hit".
EVAL_QUESTIONS = [
    # --- PURE ENGLISH QUERIES (Targeting actual PDF content) ---
    
    # Targeting: Healthcentres.pdf
    {"id": 1, "query": "Do community health centers operate mobile vans to treat patients?", "expected_keyword": "mobile vans"},
    {"id": 2, "query": "What percentage of health center patients have incomes below the federal poverty level?", "expected_keyword": "poverty"},
    {"id": 3, "query": "What types of professionals work at health centers besides doctors?", "expected_keyword": "pediatricians"},
    
    # Targeting: RHCH_Payment systems.pdf
    {"id": 4, "query": "What populations must Federally Qualified Health Centers (FQHCs) serve?", "expected_keyword": "migrant farmworkers"},
    {"id": 5, "query": "Are FQHCs required to offer free or reduced-cost care?", "expected_keyword": "reduced-cost"},
    
    # Targeting: RHCH_Info.pdf
    {"id": 6, "query": "How are Intensive Outpatient Program (IOP) payment rates determined?", "expected_keyword": "intensive outpatient"},
    {"id": 7, "query": "What are the rules for Medicare Part B vaccines and administration?", "expected_keyword": "vaccines"},
    
    # Targeting: RHCH.pdf
    {"id": 8, "query": "What qualifies an area as non-urbanized for a Rural Health Clinic?", "expected_keyword": "census bureau"},
    {"id": 9, "query": "Can a clinic be Medicare-approved as an RHC and an FQHC at the same time?", "expected_keyword": "concurrently"},
    {"id": 10, "query": "Are the services of registered dietitians covered by Rural Health Clinics?", "expected_keyword": "dietitians"},

    # --- PURE STANDARD MALAY QUERIES (Testing Cross-Lingual RAG) ---
    # Note: Expected keywords remain in English because the PDFs are in English!
    
    # Targeting: Healthcentres.pdf
    {"id": 11, "query": "Adakah pusat kesihatan menggunakan kenderaan bergerak untuk merawat pesakit?", "expected_keyword": "mobile vans"},
    {"id": 12, "query": "Berapakah peratusan pesakit pusat kesihatan yang berpendapatan di bawah paras kemiskinan persekutuan?", "expected_keyword": "poverty"},
    {"id": 13, "query": "Selain doktor, apakah pakar lain yang ada di pusat kesihatan komuniti?", "expected_keyword": "pediatricians"},
    
    # Targeting: RHCH_Payment systems.pdf
    {"id": 14, "query": "Siapakah populasi sasaran utama Pusat Kesihatan Berkelayakan Persekutuan (FQHC)?", "expected_keyword": "migrant farmworkers"},
    {"id": 15, "query": "Adakah klinik FQHC wajib memberikan rawatan kos rendah kepada individu berpendapatan rendah?", "expected_keyword": "reduced-cost"},
    
    # Targeting: RHCH_Info.pdf
    {"id": 16, "query": "Bagaimanakah kadar pembayaran program pesakit luar intensif (IOP) ditentukan?", "expected_keyword": "intensive outpatient"},
    {"id": 17, "query": "Apakah peraturan pembayaran Medicare Bahagian B untuk pentadbiran vaksin?", "expected_keyword": "vaccines"},
    
    # Targeting: RHCH.pdf
    {"id": 18, "query": "Apakah syarat Jabatan Banci AS untuk mengklasifikasikan kawasan sebagai bukan bandar?", "expected_keyword": "census bureau"},
    {"id": 19, "query": "Bolehkah sesebuah klinik mendapat kelulusan serentak sebagai RHC dan FQHC?", "expected_keyword": "concurrently"},
    {"id": 20, "query": "Adakah rawatan oleh pakar pemakanan berdaftar (dietitian) ditanggung di Klinik Kesihatan Luar Bandar?", "expected_keyword": "dietitians"}
]

if "eval_results" not in st.session_state:
    st.session_state.eval_results = []

if st.button("🚀 Run Live Evaluation (Top-K Hit Rate)"):
    results = []
    hits = 0
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, q in enumerate(EVAL_QUESTIONS):
        status_text.text(f"Evaluating query {i+1}/{len(EVAL_QUESTIONS)}: {q['query']}")
        
        # Call your RAG function
        # (Assuming it returns a dict with 'grounded_answer' and 'source_snippets')
        response = answer_query(q["query"])
        
        # Combine all retrieved source snippets into one giant string to check
        all_sources = " ".join([doc.get("snippet", "").lower() for doc in response.get("source_snippets", [])])
        
        # Calculate Top-K Hit Rate (Did the RAG pull the right document?)
        is_hit = q["expected_keyword"].lower() in all_sources
        if is_hit:
            hits += 1
            
        results.append({
            "ID": q["id"],
            "Query": q["query"],
            "Retrieved Answer": response.get("grounded_answer", "Error"),
            "Top-K Hit?": "✅ Yes" if is_hit else "❌ No",
            "Supported by Sources?": False # The manual checkbox starts as False
        })
        
        progress_bar.progress((i + 1) / len(EVAL_QUESTIONS))
        time.sleep(1) # Prevent hitting API rate limits
        
    st.session_state.eval_results = results
    
    hit_rate = (hits / len(EVAL_QUESTIONS)) * 100
    st.success(f"Evaluation Complete! Top-K Retrieval Hit Rate: {hit_rate:.1f}%")

# The Checkboxes Editor for Judges
if st.session_state.eval_results:
    st.subheader("Human Validation Step")
    st.write("Please review the generated answers and check the box if it is supported by the sources without hallucination.")
    
    df = pd.DataFrame(st.session_state.eval_results)
    
    # st.data_editor makes the boolean column an actual clickable checkbox!
    edited_df = st.data_editor(
        df,
        column_config={
            "Supported by Sources?": st.column_config.CheckboxColumn(
                "Supported? (Manual Verify)",
                help="Tick this if the AI answer is factual.",
                default=False,
            )
        },
        disabled=["ID", "Query", "Retrieved Answer", "Top-K Hit?"],
        hide_index=True,
        use_container_width=True
    )
    
    # Calculate Human Approval Rate live
    human_approved = edited_df["Supported by Sources?"].sum()
    st.metric(label="Human Verification Score", value=f"{(human_approved / len(df)) * 100:.1f}%")