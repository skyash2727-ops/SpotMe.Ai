from pydantic import BaseModel, HttpUrl, field_validator
from fastapi import FastAPI, BackgroundTasks, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import embedding_generate
import save_to_database
import numpy as np
import download
import retrive
import cv2
import re
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_methods=["*"],
    allow_headers=["*"],
)

class DriveFolderRequest(BaseModel):
    folder_url: HttpUrl
    @field_validator('folder_url')
    @classmethod
    def validate_drive_url(cls, value: HttpUrl):
        url_str = str(value)
        if value.host != "drive.google.com":
            raise ValueError("The URL must belong to drive.google.com")
        if "/folders/" not in url_str and "id=" not in url_str:
            raise ValueError("The URL does not appear to be a Google Drive folder link")
        match = re.search(r'[-\w]{25,}', url_str)
        if not match:
            raise ValueError("Could not find a valid Google Drive ID in the URL")
        return value


def process_images_in_background(url_str: str):
    print(f"\n Starting background job for URL: {url_str}")
    try:
        # 1. Start the download stream
        for file_id, img_array in download.image_stream_from_drive(url_str, batch_size=50, max_workers=8):
            print(f" Successfully downloaded image into RAM: {file_id}")
            
            # 2. Extract faces
            emb = embedding_generate.extract_embeddings(img_array)
            print(f" Found {len(emb)} faces in this image.")
            
            # 3. Save to database
            for face_embedding in emb:
                save_to_database.save_embedding_to_db(file_id, face_embedding)
                
        print("🎉 Background job completely finished!")
        
    except Exception as e:
        print(f"\n FATAL ERROR IN BACKGROUND TASK: {e}\n")
@app.post("/upload_link/")
async def process_drive_folder(request: DriveFolderRequest, background_tasks: BackgroundTasks):
    url_str = str(request.folder_url)
    background_tasks.add_task(process_images_in_background, url_str)
    return {
        "status": "success",
        "message": "Folder URL accepted! Downloading and embedding generation started in the background.",
        "url": url_str
    }

@app.post("/find-matches")
async def find_matches(selfie: UploadFile = File(...)):
    """
    This endpoint is specifically formatted for your React Frontend.
    It returns the 'match_summary' and 'matches' keys the UI expects.
    """
    # 1. Read the uploaded selfie
    raw = await selfie.read()
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    # 2. Extract embeddings
    embeddings = embedding_generate.extract_embeddings(img)
    if not embeddings:
        return JSONResponse(status_code=422, content={"detail": "No face detected in selfie."})

    # 3. Get matches with scores from your new retrive.py function
    # Uses the first face found in the selfie
    raw_matches = retrive.retrieve_similar_images_with_scores(embeddings[0])

    formatted_matches = []
    for file_id, score in raw_matches:
        formatted_matches.append({
            "photo_id": file_id,
            "drive_url": f"https://drive.google.com/thumbnail?id={file_id}&sz=w600"
        })

    return {
        "match_summary": {
            "total_matches": len(formatted_matches)
        },
        "matches": formatted_matches
    }