
import pandas as pd
import faiss
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import os

# Load model
model = SentenceTransformer('distiluse-base-multilingual-cased-v1')

# Add this BEFORE loading the index to remove old index
INDEX_PATH = "product_index.faiss"
if os.path.exists(INDEX_PATH):
    os.remove(INDEX_PATH)

# Load product data
DATA_PATH = r"products.csv"

# Load new index
INDEX_PATH = "product_index.faiss"

try:
    df = pd.read_csv(DATA_PATH, encoding='utf-8')
except UnicodeDecodeError:
    df = pd.read_csv(DATA_PATH, encoding='ISO-8859-1')

# Encode product names
product_embeddings = model.encode(df['product_name'].tolist(), convert_to_numpy=True)
faiss.normalize_L2(product_embeddings)

# Build or load FAISS index
if not os.path.exists(INDEX_PATH):
    index = faiss.IndexFlatIP(product_embeddings.shape[1])
    index.add(product_embeddings)
    faiss.write_index(index, INDEX_PATH)
else:
    index = faiss.read_index(INDEX_PATH)

# FastAPI app
app = FastAPI()

class SearchRequest(BaseModel):
    query: str
    top_k: int = 1

def search_product(query, top_k=1):
    query_vec = model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_vec)
    D, I = index.search(query_vec, top_k)
    results = df.iloc[I[0]]
    return results[['product_name', 'product_code']]

@app.post("/search")
def search(req: SearchRequest):
    result = search_product(req.query, req.top_k)
    return result.to_dict(orient="records")
