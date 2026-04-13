from insightface.app import FaceAnalysis
import numpy as np
# import download
# import save_to_database
# Initialize the model (same as before)
app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0)

# 1. We change the parameter from 'img_path' to 'img' (the numpy array)
def extract_embeddings(img: np.ndarray) -> list[list[float]]:
    """
    Takes a pre-loaded numpy array and returns a list of normalized face embeddings.
    """
    # 2. We completely remove the HEIC and cv2.imread logic! 
    # The image is ALREADY a numpy array by the time it gets here.
    # 3. Go straight to face detection
    faces = app.get(img)
    embeddings = []
    for face in faces:
        if face.det_score < 0.6:  
            continue
        embeddings.append(face.normed_embedding.tolist())
        
    return embeddings
# Assuming image_stream_from_drive is imported/defined
# FOLDER_ID = 'https://drive.google.com/drive/folders/1QU-x83Tcwj5qBMbkelCeTyf4gEaNyaZx'

# # The Drive pipeline yields the ID and the numpy array (img) directly into RAM
# if __name__ == "__main__":
#  for file_id, img in download.image_stream_from_drive(FOLDER_ID):
#     # Pass the numpy array directly into your updated InsightFace function
#     face_embeddings = extract_embeddings(img)
#     if not face_embeddings:
#         print(f"No faces found in Drive file: {file_id}")
#         continue
#     # Now you have your embeddings and your unique file_id!
#     # print(f"Extracted {len(face_embeddings)} faces from {file_id}")
#     # Example database save operation
#     for faces in face_embeddings:
#       save_to_database.save_embedding_to_db(file_id, faces)
    
