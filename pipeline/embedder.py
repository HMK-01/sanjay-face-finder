"""
ArcFace Face Embedding Generator Module
Translates visual features into 512-dimensional vector mathematics.
"""
import numpy as np
import cv2
# 🎯 CRITICAL: This import halts embedder execution until matcher finishes downloading!
from pipeline.matcher import app 

class FaceEmbedder:
    def __init__(self):
        # Bind to the globally verified single model instance
        self.app = app

    def extract_embedding(self, image: np.ndarray) -> np.ndarray:
        """
        Takes an image canvas array, uses ArcFace's high-accuracy internal 
        alignment to locate the primary face, and extracts its 512-D vector.
        """
        if image is None or image.size == 0:
            raise ValueError("Empty image matrix array passed to the embedder engine.")
            
        # InsightFace processes underlying arrays in RGB format standard
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        faces = self.app.get(rgb_image)
        if not faces:
            raise ValueError("AI Engine failed to compute facial metrics. Ensure the face is clear and well-lit.")
                
        # Return the 512-dimensional mathematical tracking array of the first detected face
        return faces[0].embedding