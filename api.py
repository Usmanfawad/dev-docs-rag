from fastapi import FastAPI

from response_generator import answer_query

app = FastAPI()


@app.get("/query")
def query_fastapi_docs(query: str, category: str = "fastapi"):
    response = answer_query(category, query)
    return {"response": response}
