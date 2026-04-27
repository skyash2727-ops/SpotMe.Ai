from insightface.app import FaceAnalysis
import numpy as np

app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0)

def extract_embeddings(img: np.ndarray) -> list[list[float]]:
    """
    Takes a pre-loaded numpy array and returns a list of normalized face embeddings.
    """
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
#     face_embeddings = extract_embeddings(img)
#     if not face_embeddings:
#         print(f"No faces found in Drive file: {file_id}")
#         continue
#     # print(f"Extracted {len(face_embeddings)} faces from {file_id}")
#     # Example database save operation
#     for faces in face_embeddings:
#       save_to_database.save_embedding_to_db(file_id, faces)
    
