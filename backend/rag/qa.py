# backend/rag/qa.py

from config import client


def generate_answer(query, context):
    prompt = f"""
    Answer the question based only on the context below. 
    If the answer is not present, say "Not found in the document". 

    Context: {context}

    Question: {query}
    """

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role":"user","content":prompt}
        ]
    )
    return response.choices[0].message.content