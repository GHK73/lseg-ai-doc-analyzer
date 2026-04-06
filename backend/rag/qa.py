# backend/rag/qa.py

from config import client


MAX_CONTEXT_CHARS = 4000  # 🔴 control context size


def _prepare_context(chunks):
    """
    chunks: list[str]
    Ensures deterministic ordering + formatting
    """

    # ensure stable ordering (important)
    chunks = list(chunks)

    # join with separators (prevents merging confusion)
    context = "\n\n---\n\n".join(chunks)

    # hard limit (avoid truncation randomness)
    if len(context) > MAX_CONTEXT_CHARS:
        context = context[:MAX_CONTEXT_CHARS]

    return context.strip()


def build_prompt(query, context):
    return f"""
    You are a financial document assistant.

    STRICT RULES:
    - Use ONLY the provided context
    - If answer is not explicitly present → say: "Not found in the document"
    - Be concise and precise
    - Do NOT hallucinate

    Context:
    {context}

    Question:
    {query}

    Answer:
    """.strip()


def generate_answer(query, retrieved_chunks):
    if not retrieved_chunks:
        return "Not found in the document"

    context = _prepare_context(retrieved_chunks)

    if not context:
        return "Not found in the document"

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        temperature=0,
        max_tokens=300,
        messages=[
            {
                "role": "system",
                "content": "Answer strictly using provided financial document context."
            },
            {
                "role": "user",
                "content": build_prompt(query, context)
            }
        ]
    )

    answer = response.choices[0].message.content.strip()

    return answer if answer else "Not found in the document"