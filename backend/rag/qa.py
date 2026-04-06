# backend/rag/qa.py

from config import client


def build_prompt(query, context):
    return f"""
    You are a financial document assistant.

    STRICT RULES:
    - Answer ONLY from the provided context
    - Do NOT guess or add external knowledge
    - If answer is not present → say: "Not found in the document"

    Context:
    {context}

    Question:
    {query}

    Answer:
    """


def generate_answer(query, context):
    prompt = build_prompt(query, context)

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        temperature=0,              # deterministic
        max_tokens=300,             # control cost + latency
        messages=[
            {"role": "system", "content": "You answer strictly from given financial documents."},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content.strip()