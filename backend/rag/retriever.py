# backend/retriver.py

import numpy as np 
def retrieve(query,model,index,chunks ,k=3):
    query_embedding = model.encode([query])
    query_embedding = query_embedding / np.linlag.norm(query_embedding, axis = 1, keepdims= True)
    distances, indices = index.search(query_embedding, k)
    results = [chunks[i] for i in indices[0]]
    return results
