import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from fastapi import FastAPI, HTTPException


# Initialize models and data
def load_models():
    # Load a pre-trained Sentence Transformer model
    sentence_model = SentenceTransformer("all-mpnet-base-v2")
    # Load a pre-trained model for text generation
    text_generator = pipeline("text-generation", model="gpt2")
    return sentence_model, text_generator


# Prepare document data and create Faiss indices
def prepare_documents():
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
    return documents


def create_indices(sentence_model, documents):
    indices = {}
    for category, docs in documents.items():
        embeddings = sentence_model.encode(docs)
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        indices[category] = (index, docs)
    return indices


def retrieve(indices, category, query, top_k=5):
    if category not in indices:
        return []

    index, docs = indices[category]
    query_embedding = sentence_model.encode([query])
    distances, indices = index.search(query_embedding, top_k)
    return [docs[i] for i in indices[0]]


def generate_response(generator, context, query, temperature=0.7):
    input_text = f"Context: {context}\n\nQuery: {query}\n\nAnswer:"
    response = generator(
        input_text, max_length=150, num_return_sequences=1, temperature=temperature
    )
    return response[0]["generated_text"]


def answer_query(indices, generator, category, query):
    retrieved_docs = retrieve(indices, category, query)
    if not retrieved_docs:
        return "The query is not related to the provided documentation category."

    context = " ".join(
        retrieved_docs[:2]
    )  # Use the top 2 most relevant documents for context
    response = generate_response(generator, context, query)
    return response


def create_app():
    app = FastAPI()

    @app.on_event("startup")
    def startup_event():
        global sentence_model, text_generator, document_indices
        sentence_model, text_generator = load_models()
        documents = prepare_documents()
        document_indices = create_indices(sentence_model, documents)

    @app.get("/query")
    def query_docs(category: str, query: str):
        if category not in document_indices:
            raise HTTPException(status_code=400, detail="Invalid category provided.")

        response = answer_query(document_indices, text_generator, category, query)
        return {"response": response}

    return app


app = create_app()
