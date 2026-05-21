"""
Face Detection Module using MediaPipe
Locates human faces within a raw image canvas.
"""
import cv2
import numpy as np
import mediapipe as mp

class FaceDetector:
    def __init__(self):
        # Initialize MediaPipe's underlying Face Detection solution
        self.mp_face_detection = mp.solutions.face_detection
        self.detector = self.mp_face_detection.FaceDetection(
            model_selection=1,  # 1 means a full-range model optimized for faces within 2-5 meters
            min_detection_confidence=0.6  # Filters out weak, uncertain false-positives
        )

    def detect_faces(self, frame: np.ndarray) -> list[dict]:
        """
        Scans an image frame and returns a list of bounding boxes for all detected faces.
        Converts coordinates from raw percentages to absolute pixel integers.
        """
        h, w, _ = frame.shape
        
        # MediaPipe requires images in RGB format, but OpenCV reads them as BGR
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.detector.process(rgb_frame)
        
        detected_faces = []
        if not results.detections:
            return detected_faces  # Return empty list if no faces are visible

        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            
            # Map normalized percentages back to real pixel coordinate values
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            width = int(bbox.width * w)
            height = int(bbox.height * h)
            
            # Guard rails: Keep boundaries inside the actual frame limits
            x, y = max(0, x), max(0, y)
            width, height = min(w - x, width), min(h - y, height)
            
            detected_faces.append({
                "box": (x, y, width, height),
                "confidence": float(detection.score[0])
            })
            
        return detected_faces