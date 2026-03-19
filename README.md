# MedBridge AI

**MedBridge AI** is a dialect-aware health information assistant built for rural and underserved communities. It helps users ask questions in local Malaysian dialects, converts them into standard language, retrieves answers from official health documents, simplifies the explanation, and provides clear next steps.

---

## Problem

Many rural users struggle to access health information because:

- official information is often written in formal or technical language
- users may speak local dialects instead of standard national language
- digital health systems are rarely designed for low digital literacy users

This creates an information gap that can delay care, reduce trust, and make important services harder to access.

---

## Solution

MedBridge AI is a website-based assistant that combines:

- **Dialect-aware input**
- **Voice and text interaction**
- **Retrieval-Augmented Generation (RAG)**
- **Plain-language simplification**
- **Action-oriented summaries**

The goal is not just to translate, but to help users **understand and act on official health information safely**.

---

## Key Features

### 1. Dialect Mode
Users can ask questions in supported local dialects.

**Current preliminary MVP support:**
- Kelantan Malay
- Sabah Malay

The system interprets dialect input into standard Malay before retrieval, then can optionally rewrite the reply back into dialect for easier communication.

### 2. Voice + Text Input
Users can either:
- type their question
- use audio input for transcription

This supports more inclusive access for users with lower literacy or lower digital confidence.

### 3. Grounded Health Answers
MedBridge AI retrieves information from official health-related documents and answers using retrieved context only.

This reduces hallucination risk and improves trust.

### 4. Easy Explanation
After retrieving the official answer, the system rewrites it into simpler everyday language.

### 5. What You Should Do Next
The system produces a short 3–5 bullet summary of next steps so users can act immediately.

### 6. Source Transparency
The interface shows the source snippets used to generate the answer.

---

## Why This Matters

This project is designed around the core idea that health information should be understandable and accessible, even for users who do not speak in standard formal language.

Instead of building a generic chatbot, MedBridge AI focuses on:

- reducing language and dialect barriers
- making official health information easier to understand
- helping users take action safely
- supporting more inclusive access to health services

---

## Prototype Scope

This preliminary-round prototype demonstrates the full flow:

1. User enters a question by voice or text
2. Optional dialect mode converts the input into standard Malay
3. The system retrieves relevant chunks from official documents
4. It generates a grounded answer
5. It simplifies the answer
6. It provides 3–5 action steps
7. It optionally rewrites the answer in dialect

---

## Current Knowledge Base

The current prototype uses health and clinic-related public documents for retrieval, including:

- rural health clinic information
- health center patient guidance
- clinic service and payment references

These documents are used to demonstrate the RAG pipeline and can later be replaced with a larger Malaysian public health document set.

---

## System Architecture

```text
Voice/Text Input
        ↓
Dialect Normalization
        ↓
RAG Retrieval from Official Documents
        ↓
Grounded Answer Generation
        ↓
Plain-Language Simplification
        ↓
3–5 Action Steps
        ↓
Optional Dialect Reply
```

---

## Tech Stack

### Frontend
- Streamlit

### Backend / AI Pipeline
- Python
- LangChain
- FAISS
- Google Gemini API

### NLP / Speech
- RapidFuzz
- Faster-Whisper

### Data
- PDF-based official health documents
- Custom dialect phrasebank

---

## Data Strategy

Our data strategy focuses on two layers:

### 1. Official document grounding
We index health-related PDF documents and retrieve relevant chunks before answer generation.

### 2. Dialect phrasebank
We maintain a structured phrasebank containing:
- dialect utterance
- standard Malay interpretation
- English meaning
- intent label

This supports both:
- dialect-to-standard normalization
- optional dialect-style reply generation

---

## Evaluation

We evaluate the system using:

- retrieval hit rate on sample questions
- manual checking of whether answers are supported by retrieved sources
- dialect interpretation quality on phrasebank examples
- end-to-end demo testing across key use cases

This keeps validation transparent and realistic for a preliminary-round prototype.

---

## Demo Use Cases

Examples of supported questions:

- “What services are available at the clinic?”
- “Can I get a vaccine here?”
- “What should I bring before visiting?”
- “Is mental health support available?”
- “Can this be done through telehealth?”

---

## Repository Structure

```text
medbridge-ai/
├── app.py
├── requirements.txt
├── README.md
├── .env.example
├── test_app.py
├── data/
│   ├── docs/
│   ├── dialect/
│   └── index/
└── medbridge/
    ├── __init__.py
    ├── ingest.py
    ├── rag.py
    ├── dialect.py
    ├── audio.py
    ├── prompts.py
    └── eval.py
```

---

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/RealdoLim/medbridge-ai.git
cd medbridge-ai
```

### 2. Create a virtual environment
```bash
python -m venv .venv
```

Activate it:

**Windows**
```bash
.venv\Scripts\activate
```

**macOS / Linux**
```bash
source .venv/bin/activate
```

</br>If you encounter an error like "...cannot be loaded because running scripts is disabled on this system." please run the following command:
```bash
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Create `.env`
Copy `.env.example` to `.env` and add your API key:

```env
GOOGLE_API_KEY=your_api_key_here
```

### 5. Build the document index
```bash
python -m medbridge.ingest
```

### 6. Run the app
```bash
streamlit run app.py
```

---

## Future Improvements

Planned next steps include:

- larger Malaysian health document set
- more dialect support
- stronger dialect reply quality
- improved voice UX
- expanded evaluation dashboard
- deployment-ready API and frontend polish

---

## Team

Add your team member names here.

- Realdo Aginda Lim
- Ekin Lunar Limarya
- Andrian Hindra
- Ryan So

---

## Important Note

MedBridge AI is an information access assistant. It is designed to help users understand official health information more clearly. It does **not** provide medical diagnosis, emergency decision-making, or professional medical advice.

---
