from transformers import pipeline

from sentence_transformer import retrieve

# Load a pre-trained model for text generation
generator = pipeline("text-generation", model="gpt2")


def generate_response(context, query, temperature=0.7):
    input_text = f"Context: {context}\n\nQuery: {query}\n\nAnswer:"
    response = generator(
        input_text, max_length=150, num_return_sequences=1, temperature=temperature
    )
    return response[0]["generated_text"].split("Answer:")[-1].strip()


def answer_query(category, query):
    retrieved_docs = retrieve(category, query)
    if not retrieved_docs:
        return "The query is not related to the provided documentation category."

    context = " ".join(
        retrieved_docs[:2]
    )  # Use the top 2 most relevant documents for context
    response = generate_response(context, query)
    return response
