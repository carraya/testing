from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel
from deposition_vectorizer import DepositionVectorizer
import os
from dotenv import load_dotenv
from fastapi.responses import FileResponse
from fastapi.testclient import TestClient

load_dotenv()

app = FastAPI()

PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
TOGETHERAI_API_KEY = os.getenv("TOGETHERAI_API_KEY")

deposition_vectorizer = DepositionVectorizer(
    pinecone_index_name=PINECONE_INDEX_NAME,
    pinecone_api_key=PINECONE_API_KEY,
    togetherai_api_key=TOGETHERAI_API_KEY,
)

class VideoParams(BaseModel):
    window: int = 5
    overlap: int = 1

@app.post("/vectorize_and_upsert_pdf/")
async def vectorize_and_upsert_pdf(file: UploadFile = File(...)):
    try:
        file_location = f"temp/{file.filename}"
        with open(file_location, "wb+") as file_object:
            file_object.write(file.file.read())
        deposition_vectorizer.vectorize_and_upsert_pdf(file_location)
        return {"message": "PDF processed successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/vectorize_and_upsert_video/")
async def vectorize_and_upsert_video(video_params: VideoParams, file: UploadFile = File(...)):
    try:
        file_location = f"temp/{file.filename}"
        with open(file_location, "wb+") as file_object:
            file_object.write(file.file.read())
        deposition_vectorizer.vectorize_and_upsert_video(file_location, video_params.window, video_params.overlap)
        return {"message": "Video processed successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class QueryText(BaseModel):
    text: str
    top_k: int = 3
    filter: dict = None

@app.post("/query_text/")
async def query_text(query: QueryText):
    try:
        result = deposition_vectorizer.query_text(query.text, query.top_k, query.filter)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

from deposition_matcher import DepositionMatcher

deposition_matcher = DepositionMatcher(
    pinecone_index_name=PINECONE_INDEX_NAME,
    pinecone_api_key=PINECONE_API_KEY,
    togetherai_api_key=TOGETHERAI_API_KEY,
)

class QueryVideo(BaseModel):
    video_path: str
    max_doc: int = 5
    max_window: int = 3

@app.post("/query_video/")
async def query_video(query: QueryVideo):
    try:
        result = deposition_matcher.query(query.video_path, query.max_doc, query.max_window)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))