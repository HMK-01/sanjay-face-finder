"""
Cosine Similarity Vectors Matching Engine
Performs direct matrix multiplication to find geometric distances between faces.
"""
import numpy as np
from config import SIMILARITY_THRESHOLD

class EmbeddingMatcher:
    @staticmethod
    def cosine_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Computes the raw cosine similarity score between two vector tracks."""
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(dot_product / (norm1 * norm2))

    @staticmethod
    def find_best_match(target_embedding: np.ndarray, registry: dict[str, list[np.ndarray]]) -> tuple[str, float]:
        """
        Compares a face embedding against the registered memory store.
        """
        best_name = "Unknown"
        best_score = -1.0

        for person_name, registered_embeddings in registry.items():
            for ref_embedding in registered_embeddings:
                score = EmbeddingMatcher.cosine_similarity(target_embedding, ref_embedding)
                
                if score > best_score:
                    best_score = score
                    best_name = person_name

        # Enforce threshold fallback check
        if best_score < SIMILARITY_THRESHOLD:
            return "Unknown", 0.0

        return best_name, best_score