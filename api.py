from fastapi import FastAPI, HTTPException, File, UploadFile
from deposition_vectorizer import DepositionVectorizer
from pydantic import BaseModel
import os

app = FastAPI()

# Assuming environment variables are already set for API keys and Pinecone index name
pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
togetherai_api_key = os.getenv("TOGETHERAI_API_KEY")

deposition_vectorizer = DepositionVectorizer(
    pinecone_index_name=pinecone_index_name,
    pinecone_api_key=pinecone_api_key,
    togetherai_api_key=togetherai_api_key,
)

class QueryTextRequest(BaseModel):
    text: str
    top_k: int = 3
    filter: dict = None

@app.post("/vectorize-and-upsert-pdf/")
async def vectorize_and_upsert_pdf(pdf_file: UploadFile = File(...)):
    try:
        temp_file_path = f"temp_{pdf_file.filename}"
        with open(temp_file_path, "wb") as buffer:
            buffer.write(pdf_file.file.read())
        deposition_vectorizer.vectorize_and_upsert_pdf(temp_file_path)
        os.remove(temp_file_path)
        return {"message": "PDF processed and vectors upserted successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/vectorize-and-upsert-video/")
async def vectorize_and_upsert_video(video_file: UploadFile = File(...)):
    try:
        temp_file_path = f"temp_{video_file.filename}"
        with open(temp_file_path, "wb") as buffer:
            buffer.write(video_file.file.read())
        deposition_vectorizer.vectorize_and_upsert_video(temp_file_path)
        os.remove(temp_file_path)
        return {"message": "Video processed and vectors upserted successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query-text/")
async def query_text(query: QueryTextRequest):
    try:
        results = deposition_vectorizer.query_text(query.text, top_k=query.top_k, filter=query.filter)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
