# backend/rag/qa.py

from config import get_client


MAX_CONTEXT_CHARS = 4000



def _prepare_context(chunks):
    """
    Prioritize high-signal chunks instead of naive truncation
    """

    context_parts = []
    total_len = 0

    for chunk in chunks:
        chunk_len = len(chunk)

        if total_len + chunk_len > MAX_CONTEXT_CHARS:
            break

        context_parts.append(chunk)
        total_len += chunk_len

    context = "\n\n---\n\n".join(context_parts)

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
    client = get_client()
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