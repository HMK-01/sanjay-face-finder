"""
Face Detector Module
Locates bounding boxes of human faces within video streams.
"""
# 🎯 Share the identical engine instance from matcher
from pipeline.matcher import app

class FaceDetector:
    def __init__(self):
        self.engine = app

    def detect_faces(self, frame):
        """Locates boundaries of all visible facial profiles in a frame"""
        if frame is None or frame.size == 0:
            return []
            
        import cv2
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = self.engine.get(rgb_frame)
        
        processed_detections = []
        for face in faces:
            # Convert internal bbox coordinates to standard dictionary format for app.py loop
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
            processed_detections.append({
                "box": [x1, y1, x2 - x1, y2 - y1],
                "confidence": face.det_score
            })
            
        return processed_detections