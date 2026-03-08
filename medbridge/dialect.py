from __future__ import annotations

import re
from typing import Optional

import pandas as pd
from rapidfuzz import fuzz
from langchain_google_genai import ChatGoogleGenerativeAI

def _clean_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_to_standard(
    text: str,
    dialect: str,
    phrasebank_df: pd.DataFrame,
    threshold: int = 80,
) -> str:
    """
    Convert a dialect user query into standard Malay using phrasebank fuzzy matching.
    If no strong match is found, return the original text.
    """
    if not text:
        return text

    cleaned_text = _clean_text(text)
    dialect = dialect.lower().strip()

    subset = phrasebank_df[phrasebank_df["dialect"].str.lower() == dialect].copy()

    if subset.empty:
        return text

    best_score = -1
    best_standard = text

    for _, row in subset.iterrows():
        candidate = _clean_text(str(row["utterance"]))
        score = fuzz.ratio(cleaned_text, candidate)

        if score > best_score:
            best_score = score
            best_standard = str(row["standard_malay"])

    if best_score >= threshold:
        return best_standard

    return text


def paraphrase_to_dialect(
    text: str,
    dialect: str,
    phrasebank_df: pd.DataFrame,
) -> str:
    """
    Light MVP dialect styling:
    replace known standard Malay phrases with dialect utterances.
    """
    if not text:
        return text

    dialect = dialect.lower().strip()
    styled_text = text

    subset = phrasebank_df[phrasebank_df["dialect"].str.lower() == dialect].copy()

    if subset.empty:
        return text

    # Longer replacements first to avoid partial replacement problems
    subset["std_len"] = subset["standard_malay"].astype(str).str.len()
    subset = subset.sort_values("std_len", ascending=False)

    for _, row in subset.iterrows():
        standard_phrase = str(row["standard_malay"]).strip()
        dialect_phrase = str(row["utterance"]).strip()

        if standard_phrase and standard_phrase.lower() in styled_text.lower():
            pattern = re.compile(re.escape(standard_phrase), re.IGNORECASE)
            styled_text = pattern.sub(dialect_phrase, styled_text)

    return styled_text

def rewrite_fully_to_dialect(
    text: str,
    dialect: str,
    phrasebank_df: pd.DataFrame,
) -> str:
    """
    Rewrite the whole text naturally into the chosen dialect.
    Uses phrasebank examples as style guidance.
    Falls back to light phrase replacement if needed.
    """
    if not text:
        return text

    dialect = dialect.lower().strip()

    subset = phrasebank_df[phrasebank_df["dialect"].str.lower() == dialect].copy()
    if subset.empty:
        return text

    examples = []
    for _, row in subset.head(12).iterrows():
        examples.append(
            f"Standard Malay: {row['standard_malay']}\n"
            f"{dialect.title()} dialect: {row['utterance']}"
        )

    examples_text = "\n\n".join(examples)

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.2
    )

    prompt = f"""
You are a dialect rewriting assistant.

Task:
Rewrite the text fully and naturally into {dialect.title()} dialect.

Rules:
1. Preserve the exact medical meaning.
2. Do not add facts.
3. Do not remove facts.
4. Keep headings, bullet points, and structure.
5. Make the final output sound naturally like {dialect.title()} dialect, not just a few replaced words.
6. Do not have medical jargon, replace with simpler words but still have the exact medical meaning.
7. Return only the rewritten text.

Phrase examples:
{examples_text}

Text to rewrite:
{text}
"""

    try:
        response = llm.invoke(prompt)
        rewritten = response.content if hasattr(response, "content") else str(response)
        rewritten = rewritten.strip()

        if rewritten:
            return rewritten

    except Exception:
        pass

    return paraphrase_to_dialect(text, dialect, phrasebank_df)