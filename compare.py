import numpy as np
def compare_by_cosine_similarity(normed_emb1: list[float], normed_emb2: list[float]) -> float:
    vec1 = np.array(normed_emb1)
    vec2 = np.array(normed_emb2)
    similarity = np.dot(vec1, vec2)
    return float(similarity)