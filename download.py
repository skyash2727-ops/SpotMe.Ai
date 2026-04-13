from googleapiclient.discovery import build
import requests
import cv2
import numpy as np
import re
from concurrent.futures import ThreadPoolExecutor
from googleapiclient.errors import HttpError
API_KEY = "AIzaSyDLWU6rBATbhy0XMnmpc4iR5gMWw1UJQWw"
drive_service = build(
    "drive",
    "v3",
    developerKey=API_KEY
)
def extract_folder_id(folder_url):
    match = re.search(r'folders/([a-zA-Z0-9_-]+)', folder_url)
    if match:
        return match.group(1)
    
    raise ValueError("Invalid folder link")


def get_folder_items(folder_id):
    items = []
    page_token = None
    try:
        while True:
            response = drive_service.files().list(
                q=f"'{folder_id}' in parents",
                fields="nextPageToken, files(id,name,mimeType)",
                pageSize=1000,
                pageToken=page_token
            ).execute()
            
            items.extend(response.get("files", []))
            page_token = response.get("nextPageToken")
            if not page_token:
                break

    except HttpError as e:
        if e.resp.status in [403, 404]:
            raise Exception("Drive folder is private or inaccessible")

        raise e

    return items

def get_all_image_ids(folder_id):
    file_ids = []
    items = get_folder_items(folder_id)
    for item in items:
        fid = item["id"]
        mime = item["mimeType"]
        if mime.startswith("image/"):
            file_ids.append(fid)
        elif mime == "application/vnd.google-apps.folder":
            file_ids.extend(get_all_image_ids(fid))

    return file_ids


def download_image(file_id):
    url = f"https://www.googleapis.com/drive/v3/files/{file_id}?alt=media&key={API_KEY}"
    response = requests.get(url, timeout = 15)
    if response.status_code != 200:
        return None
    def download_and_preprocess(fid):
        img = download_image(fid)
        if img is None:
            print(f"❌ Failed to download or decode image ID: {fid}") # ADD THIS LINE
        return None
    image_bytes = np.asarray(bytearray(response.content), dtype=np.uint8)
    img = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

    return img


def preprocess_image(img, max_size=800):
    h, w = img.shape[:2]
    scale = max_size / max(h, w)
    if scale < 1:
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = cv2.resize(img, (new_w, new_h))

    return img
def download_and_preprocess(fid):
    img = download_image(fid)
    if img is None:
        return None
    img = preprocess_image(img)
    
    return img

def batch_generator(items, batch_size=50):
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def image_stream_from_drive(folder_url, batch_size=100, max_workers=8):
    folder_id = extract_folder_id(folder_url)
    file_ids = get_all_image_ids(folder_id)

    for batch in batch_generator(file_ids, batch_size):
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = executor.map(download_and_preprocess, batch)

        for fid, img in zip(batch, results):
            if img is not None:
                yield fid, img