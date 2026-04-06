# backend/rag/qa.py

from config import client


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
    """


def generate_answer(query, context):
    if not context.strip():
        return "Not found in the document"

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        temperature=0,
        max_tokens=300,
        messages=[
            {"role": "system", "content": "Answer strictly using provided financial document context."},
            {"role": "user", "content": build_prompt(query, context)}
        ]
    )

    answer = response.choices[0].message.content.strip()

    # ---- Safety fallback ----
    if not answer:
        return "Not found in the document"

    return answer