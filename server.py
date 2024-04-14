from fastapi import FastAPI, File, UploadFile, HTTPException, Response
import uvicorn
import os
from whisper_func import VideoToTextModel
from pathlib import Path

app = FastAPI()
model = VideoToTextModel()


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    contents = await file.read()
    # Save the file to a specific location
    with open("data/" + file.filename, "wb") as f:
        f.write(contents)

    # Preprocess the file
    segments = []
    documents = []
    if file.filename.endswith(".mp4"):
        segments = model.sentence_transcribe("data/" + file.filename)
    elif file.filename.endswith(".pdf"):
        documents = []

    return {
        "status": "ok",
        "statusCode": 200,
        "segments": segments,
        "documents": documents,
    }


@app.delete("/delete")
async def delete_file(file: str):
    file_path = os.path.join("data", file)
    print(file_path)
    if os.path.exists(file_path):
        file_stem = Path(file_path).stem
        for file in os.listdir("data"):
            if file.startswith(file_stem):
                os.remove(os.path.join("data", file))
        # os.remove(file_path)
        return {"status": "ok", "statusCode": 200}
    else:
        raise HTTPException(status_code=404, detail="File not found")


@app.get("/files")
async def get_orig_files():
    files = [file for file in os.listdir("data") if "[" not in file]
    return {"files": files}


@app.get("/files/{filename}")
async def get_file(filename: str):
    # Only add the data folder if it is not already there
    file_path = os.path.join("data", filename)
    print("Trying to get", file_path)
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            contents = f.read()
        return Response(content=contents, media_type="application/octet-stream")
    else:
        raise HTTPException(status_code=404, detail="File not found")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=1000, reload=True)
