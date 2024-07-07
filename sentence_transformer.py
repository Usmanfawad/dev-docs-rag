import faiss
import numpy as np
from sentence_transformers import SentenceTransformer, util

# Load a pre-trained Sentence Transformer model
model = SentenceTransformer("all-mpnet-base-v2")

# Assuming 'documents' is a list of strings where each string is a section from FastAPI docs
documents = {
    "fastapi": [
        "FastAPI is a modern, fast (high-performance), web framework for building APIs with Python 3.6+ based on standard Python type hints.",
        "To install FastAPI, you need to run: pip install fastapi[all]",
        "Creating your first FastAPI application is simple. Import FastAPI and create an instance:",
        "You can define your API endpoints using standard Python functions and type hints.",
        "SqlModels can easily be created in the format of Pydantic models.",
    ],
    "nextjs": [
        "Next.js is a React framework that enables several extra features, including server-side rendering and generating static websites.",
        "To install Next.js, you need to run: npm install next react react-dom",
        "Creating your first Next.js application is simple. Create a pages directory and add index.js file:",
        "You can define your routes by adding files to the pages directory.",
        "Next.js supports API routes, which are serverless functions you can create in the pages/api directory.",
    ],
}


def retrieve(category, query, top_k=5, relevance_threshold=0.5):
    if category not in documents:
        return []

    # Encode documents and create index for the specified category
    docs = documents[category]
    embeddings = model.encode(docs)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, top_k)

    # Calculate relevance scores
    relevance_scores = util.cos_sim(query_embedding, embeddings)[0]
    relevant_docs = []
    for idx, score in zip(indices[0], relevance_scores[indices[0]]):
        if score >= relevance_threshold:
            relevant_docs.append(docs[idx])

    return relevant_docs
