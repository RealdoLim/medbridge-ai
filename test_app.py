import pytest
import streamlit as st
import pandas as pd
from unittest.mock import patch, MagicMock

import app

@pytest.fixture
def phrasebank_df():
    # Create a minimal DataFrame for dialect phrases
    return pd.DataFrame({
        'standard': ['clinic', 'vaccination', 'assistance'],
        'kelantan': ['klinik', 'vaksinasi', 'bantuan'],
        'sabah': ['klinik', 'vaksinasi', 'bantuan']
    })

def test_load_phrasebank(monkeypatch):
    # Patch pd.read_csv to return a dummy DataFrame
    monkeypatch.setattr(pd, "read_csv", lambda path: pd.DataFrame({'a': [1]}))
    df = app.load_phrasebank()
    assert isinstance(df, pd.DataFrame)

def test_get_llm_returns_instance():
    llm = app.get_llm()
    assert hasattr(llm, "invoke")

@pytest.mark.parametrize("text,target_language", [
    ("Hello", "Malay"),
    ("", "English"),
    ("not found in docs", "Malay"),
])

def test_translate_text(monkeypatch, text, target_language):
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = f"{text} translated to {target_language}"
    mock_llm.invoke.return_value = mock_response
    monkeypatch.setattr(app, "get_llm", lambda: mock_llm)
    result = app.translate_text(text, target_language)
    assert isinstance(result, str)
    if text and text != "not found in docs":
        assert target_language in result

def test_clear_text_sets_session_state(monkeypatch):
    st.session_state.user_query = "something"
    app.clear_text()
    assert st.session_state.user_query == ""

def test_dialect_normalization(monkeypatch, phrasebank_df):
    monkeypatch.setattr(app, "phrasebank_df", phrasebank_df)
    monkeypatch.setattr(app, "normalize_to_standard", lambda q, d, df: "clinic")
    result = app.normalize_to_standard("klinik", "kelantan", phrasebank_df)
    assert result == "clinic"

def test_rewrite_fully_to_dialect(monkeypatch, phrasebank_df):
    monkeypatch.setattr(app, "phrasebank_df", phrasebank_df)
    monkeypatch.setattr(app, "rewrite_fully_to_dialect", lambda ans, d, df: "klinik")
    result = app.rewrite_fully_to_dialect("clinic", "kelantan", phrasebank_df)
    assert result == "klinik"

def test_answer_query(monkeypatch):
    mock_result = {
        "grounded_answer": "Official info",
        "simplified_answer": "Simple info",
        "action_steps": "Do this",
        "source_snippets": [{"source": "doc", "page": 1, "snippet": "info"}]
    }
    monkeypatch.setattr(app, "answer_query", lambda q: mock_result)
    result = app.answer_query("clinic")
    assert isinstance(result, dict)
    assert "grounded_answer" in result

def test_transcribe_uploaded_audio(monkeypatch):
    monkeypatch.setattr(app, "transcribe_uploaded_audio", lambda audio: "clinic")
    result = app.transcribe_uploaded_audio("dummy_audio")
    assert result == "clinic"

# Run using: "pytest test_app.py"
