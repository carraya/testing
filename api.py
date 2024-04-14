from fastapi import FastAPI, HTTPException, File, UploadFile
from deposition_vectorizer import DepositionVectorizer
from pydantic import BaseModel
import os
import io
from fastapi.responses import StreamingResponse
from VideoStitcher import VideoStitcher

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

@app.post("/query-video-timestamps/")
async def query_video_timestamps(query: QueryTextRequest, video_file: UploadFile = File(...)):
    try:
        results = await deposition_vectorizer.query_text(query.text, top_k=query.top_k, filter=query.filter)
        timestamps = [{"start": result["metadata"]["start"], "end": result["metadata"]["end"]} for result in results["matches"]]
        
        # Save the uploaded video to a temporary file
        temp_video_path = f"temp_{video_file.filename}"
        with open(temp_video_path, "wb") as buffer:
            buffer.write(video_file.file.read())
        
        # Initialize VideoStitcher with the temporary video path
        video_stitcher = VideoStitcher(input_path=temp_video_path, output_path=f"processed_{video_file.filename}", captions=False)
        
        # Segment the video based on the timestamps and save the output
        video_stitcher.stitch(segments=timestamps)
        
        # Read the processed video file and return it as a blob
        with open(f"processed_{video_file.filename}", "rb") as video:
            processed_video_blob = video.read()
        
        # Clean up temporary and processed files
        os.remove(temp_video_path)
        os.remove(f"processed_{video_file.filename}")
        
        return StreamingResponse(io.BytesIO(processed_video_blob), media_type="video/mp4")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
      
