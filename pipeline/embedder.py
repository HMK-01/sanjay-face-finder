"""
ArcFace Face Embedding Generator Module
Translates visual features into 512-dimensional vector mathematics.
"""
import numpy as np
import cv2
from insightface.app import FaceAnalysis

class FaceEmbedder:
    def __init__(self):
        # Initialize InsightFace using the highly accurate buffalo_l model bundle
        self.app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
        # Prepare the structural weights matrix context completely on CPU
        self.app.prepare(ctx_id=-1, det_size=(640, 640))

    def extract_embedding(self, image: np.ndarray) -> np.ndarray:
        """
        Takes an image canvas array, uses ArcFace's high-accuracy internal 
        alignment to locate the primary face, and extracts its 512-D vector.
        """
        faces = self.app.get(image)
        if not faces:
            raise ValueError("AI Engine failed to compute facial metrics. Ensure the face is clear and well-lit.")
                
        # Return the 512-dimensional mathematical tracking array of the first detected face
        return faces[0].embedding