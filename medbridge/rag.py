from pathlib import Path

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

from medbridge.prompts import build_answer_prompt

load_dotenv()

INDEX_DIR = Path("data/index")


def load_vectorstore():
    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001"
    )

    vectorstore = FAISS.load_local(
        str(INDEX_DIR),
        embeddings,
        allow_dangerous_deserialization=True
    )
    return vectorstore


def retrieve_docs(query: str, k: int = 4):
    vectorstore = load_vectorstore()
    docs = vectorstore.similarity_search(query, k=k)
    return docs


def format_context(docs):
    context_blocks = []
    source_snippets = []

    for i, doc in enumerate(docs, start=1):
        source = Path(doc.metadata.get("source", "unknown")).name
        page = doc.metadata.get("page", None)

        if isinstance(page, int):
            page_label = f"page {page + 1}"
            page_number = page + 1
        else:
            page_label = "page unknown"
            page_number = None

        content = doc.page_content.strip()

        context_blocks.append(
            f"[Source {i} | {source} | {page_label}]\n{content}"
        )

        source_snippets.append({
            "source": source,
            "page": page_number,
            "snippet": content[:300]
        })

    return "\n\n".join(context_blocks), source_snippets


def parse_response(text: str):
    sections = {
        "grounded_answer": "",
        "simplified_answer": "",
        "action_steps": ""
    }

    current = None

    for line in text.splitlines():
        stripped = line.strip()
        lower = stripped.lower()

        if lower.startswith("grounded answer"):
            current = "grounded_answer"
            continue
        elif lower.startswith("simplified answer"):
            current = "simplified_answer"
            continue
        elif lower.startswith("action steps"):
            current = "action_steps"
            continue

        if current:
            sections[current] += line + "\n"

    for key in sections:
        sections[key] = sections[key].strip()

    return sections


def answer_query(user_query: str, k: int = 4):
    docs = retrieve_docs(user_query, k=k)
    context, source_snippets = format_context(docs)

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0
    )

    prompt = build_answer_prompt(user_query, context)
    response = llm.invoke(prompt)

    if hasattr(response, "content"):
        raw_text = response.content
    else:
        raw_text = str(response)

    parsed = parse_response(raw_text)

    return {
        "grounded_answer": parsed["grounded_answer"],
        "simplified_answer": parsed["simplified_answer"],
        "action_steps": parsed["action_steps"],
        "source_snippets": source_snippets,
        "raw_response": raw_text
    }


if __name__ == "__main__":
    query = input("Enter your question: ")
    result = answer_query(query)

    print("\nGROUNDed ANSWER:\n")
    print(result["grounded_answer"])

    print("\nSIMPLIFIED ANSWER:\n")
    print(result["simplified_answer"])

    print("\nACTION STEPS:\n")
    print(result["action_steps"])

    print("\nSOURCE SNIPPETS:\n")
    for item in result["source_snippets"]:
        print(item)