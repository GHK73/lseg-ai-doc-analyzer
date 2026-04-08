from config import get_client
import numpy as np

MAX_CONTEXT_CHARS = 4000


# -------- Context Preparation --------
def _prepare_context(retrieved_results):
    """
    retrieved_results = [
        {"text": "...", "page": 2, "score": 0.91}
    ]
    """

    context_parts = []
    total_len = 0
    sources = []

    for item in retrieved_results:
        text = item["text"]
        page = item.get("page", "N/A")

        if not text:
            continue

        chunk_len = len(text)

        if total_len + chunk_len > MAX_CONTEXT_CHARS:
            break

        context_parts.append(f"[Page {page}]\n{text}")
        sources.append(page)

        total_len += chunk_len

    context = "\n\n---\n\n".join(context_parts)

    return context.strip(), list(set(sources))


# -------- Prompt --------
def build_prompt(query, context):
    return f"""
You are an expert financial analyst.

STRICT RULES:
- Use ONLY the provided context
- If answer is not explicitly present → say: "Not found in the document"
- Cite page numbers like (Page X)
- Be concise and factual
- Do NOT hallucinate

Context:
{context}

Question:
{query}

Answer:
""".strip()


# -------- Confidence --------
def _compute_confidence(results):
    if not results:
        return 0.0

    scores = [r.get("score", 0.0) for r in results]
    return float(np.mean(scores))


# -------- Main QA --------
def generate_answer(query, retrieved_results):
    """
    Returns:
    {
        "answer": "...",
        "sources": [1, 3],
        "confidence": 0.87
    }
    """

    if not retrieved_results:
        return {
            "answer": "Not found in the document",
            "sources": [],
            "confidence": 0.0
        }

    # sort by relevance (important)
    retrieved_results = sorted(
        retrieved_results,
        key=lambda x: x.get("score", 0.0),
        reverse=True
    )

    context, sources = _prepare_context(retrieved_results)

    if not context:
        return {
            "answer": "Not found in the document",
            "sources": [],
            "confidence": 0.0
        }

    client = get_client()

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        temperature=0,
        max_tokens=300,
        messages=[
            {
                "role": "system",
                "content": "You are a strict financial QA system that never hallucinates."
            },
            {
                "role": "user",
                "content": build_prompt(query, context)
            }
        ]
    )

    answer = response.choices[0].message.content.strip()

    return {
        "answer": answer if answer else "Not found in the document",
        "sources": sources,
        "confidence": round(_compute_confidence(retrieved_results), 3)
    }