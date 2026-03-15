import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

from medbridge.audio import transcribe_uploaded_audio
from medbridge.dialect import normalize_to_standard, rewrite_fully_to_dialect
from medbridge.rag import answer_query

load_dotenv()

def clear_text():
    st.session_state.user_query = ""

st.set_page_config(page_title="MedBridge AI", layout="wide")

st.title("🏥 MedBridge AI")

st.markdown("""
            ### Your health information assistant

            You can:
            - Ask in **English, Malay, or your dialect**
            - **Type** your question or **upload voice**
            - Get **simple explanations** and **clear steps**

            Ask anything about clinics, health services, or government help.
            """)
st.markdown("<hr style='margin-top:5px;margin-bottom:10px;'>", unsafe_allow_html=True)

# UI Improvements
st.markdown("""
            <style>
            /* Main page padding */
            .block-container {
                padding-top: 2.5rem;
                padding-bottom: 1rem;
            }

            /* Title styling */
            h1 {
                font-size: 2.2rem;
                font-weight: 700;
            }
            
            /* Section headers */
            h2 {
                font-size: 1.5rem;
                margin-top: 1.5rem;
            }

            /* Card style container */
            .card {
                background-color: #1e293b;
                padding: 22px;
                border-radius: 14px;
                border: 1px solid #374151;
                margin-top: 20px;
                margin-bottom: 20px;
            }

            /* Button styling */
            .stButton>button {
                width: 100%;
                border-radius: 10px;
                font-weight: 600;
                padding: 8px;
            }

            /* Sidebar header */
            section[data-testid="stSidebar"] h2 {
                font-size: 1.2rem;
            }

                /* Expander styling */
                .streamlit-expanderHeader {
                font-weight: 600;
            }

                </style>
            """, unsafe_allow_html=True)




@st.cache_data
def load_phrasebank():
    return pd.read_csv("data/dialect/dialect_phrases.csv")


@st.cache_resource
def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0
    )


def translate_text(text: str, target_language: str) -> str:
    if not text:
        return text

    if text.strip().lower() == "not found in docs":
        return text

    llm = get_llm()

    prompt = f"""
Translate the text below into {target_language}.
Preserve the meaning, bullet points, headings, and line breaks.
Return only the translated text.

Text:
{text}
"""

    response = llm.invoke(prompt)
    return response.content if hasattr(response, "content") else str(response)


phrasebank_df = load_phrasebank()

with st.sidebar:
    st.header("Controls")

    dialect_mode = st.selectbox(
        "Dialect Mode",
        ["OFF", "Kelantan", "Sabah"]
    )

    if "reply_in_dialect" not in st.session_state:
        st.session_state.reply_in_dialect = False

    if dialect_mode == "OFF":
        st.session_state.reply_in_dialect = False

    reply_in_dialect = st.checkbox(
        "Reply in dialect",
        key="reply_in_dialect",
        disabled=(dialect_mode == "OFF")
    )

    if reply_in_dialect and dialect_mode != "OFF":
        st.markdown(f"**Output Language:** {dialect_mode} dialect")
        output_language = "Dialect"
        st.caption("Output language is locked because reply in dialect is enabled.")
        st.caption("Final answer will be rewritten fully into the selected dialect.")
    else:
        output_language = st.selectbox(
            "Output Language",
            ["Malay", "English"]
        )

st.subheader("Ask Your Question")
st.subheader("Try asking:")

col1, col2, col3 = st.columns(3)

if col1.button("Nearest clinic"):
    st.session_state.user_query = "Where is the nearest clinic?"

if col2.button("Vaccination information"):
    st.session_state.user_query = "How can I get vaccination?"

if col3.button("Government health aid"):
    st.session_state.user_query = "How do I apply for health assistance?"

input_mode = st.radio(
    "How would you like to ask?",
    ["Type your question", "Use voice recording"]
)

if input_mode == "Type your question":
    if "user_query" not in st.session_state:
        st.session_state.user_query= ""
    user_query = st.text_area(
    "Type your question here",
    placeholder="Example: mano nak gi klinik?",
    height=120,
    key = "user_query"
    )
    
    uploaded_audio = None
else:
    uploaded_audio = st.file_uploader(
        "Upload your voice recording",
        type=["wav", "mp3", "m4a"]
    )
    user_query = ""

col1, col2 = st.columns(2)
with col1:
    run_button = st.button("🔎 Get Help", type="primary")
with col2:
    clear_button = st.button("❌ Clear Question", on_click=clear_text)


if run_button:
    raw_query = ""

    if input_mode == "Type your question":
        raw_query = user_query.strip()
        if not raw_query:
            st.warning("Please enter a question first.")
            st.stop()
    else:
        if uploaded_audio is None:
            st.warning("Please upload an audio file first.")
            st.stop()

        with st.spinner("Transcribing audio..."):
            raw_query = transcribe_uploaded_audio(uploaded_audio)

        st.subheader("Transcript")
        st.write(raw_query)

        if not raw_query:
            st.warning("Could not transcribe the audio.")
            st.stop()

    interpreted_query = raw_query

    if dialect_mode != "OFF":
        interpreted_query = normalize_to_standard(
            raw_query,
            dialect_mode.lower(),
            phrasebank_df
        )

    st.subheader("We interpreted your dialect as:")

    st.markdown(f"""
                <div style="
                background:#1e293b;
                padding:16px;
                border-radius:10px;
                border:1px solid #374151;
                margin-bottom:20px;
                font-size:16px;">
                {interpreted_query}
                </div>
                """, unsafe_allow_html=True)

    if reply_in_dialect and dialect_mode != "OFF":
        st.info(f"Final answer will be rewritten into {dialect_mode} dialect.")

    with st.spinner("Searching official docs and generating answer..."):
        result = answer_query(interpreted_query)

    grounded_answer = result["grounded_answer"] or "not found in docs"
    simplified_answer = result["simplified_answer"] or "not found in docs"
    action_steps = result["action_steps"] or "- not found in docs"

    # Dialect reply overrides normal output language.
    if reply_in_dialect and dialect_mode != "OFF":
        grounded_answer = translate_text(grounded_answer, "Bahasa Melayu")
        simplified_answer = translate_text(simplified_answer, "Bahasa Melayu")
        action_steps = translate_text(action_steps, "Bahasa Melayu")

        grounded_answer = rewrite_fully_to_dialect(
            grounded_answer,
            dialect_mode.lower(),
            phrasebank_df
        )
        simplified_answer = rewrite_fully_to_dialect(
            simplified_answer,
            dialect_mode.lower(),
            phrasebank_df
        )
        action_steps = rewrite_fully_to_dialect(
            action_steps,
            dialect_mode.lower(),
            phrasebank_df
        )
    else:
        if output_language == "Malay":
            grounded_answer = translate_text(grounded_answer, "Bahasa Melayu")
            simplified_answer = translate_text(simplified_answer, "Bahasa Melayu")
            action_steps = translate_text(action_steps, "Bahasa Melayu")
        else:
            grounded_answer = translate_text(grounded_answer, "English")
            simplified_answer = translate_text(simplified_answer, "English")
            action_steps = translate_text(action_steps, "English")

    st.markdown(f"""
<div class="card">
<h3>📖 Official Information</h3>
<p>{grounded_answer}</p>
</div>
""", unsafe_allow_html=True)

    st.markdown(f"""
<div class="card">
<h3>💡 Easy Explanation</h3>
<p>{simplified_answer}</p>
</div>
""", unsafe_allow_html=True)

    st.markdown(f"""
<div class="card">
<h3>✅ What You Should Do</h3>
<p>{action_steps}</p>
</div>
""", unsafe_allow_html=True)

    st.subheader("Sources Used")
    for i, item in enumerate(result["source_snippets"], start=1):
        source = item.get("source", "unknown")
        page = item.get("page", "unknown")
        snippet = item.get("snippet", "")

        with st.expander(f"Source {i}: {source} (page {page})"):
            st.write(snippet)