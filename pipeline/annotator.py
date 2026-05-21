"""
OpenCV Frame Annotation Engine
Draws visual overlays and diagnostic labels onto matching image coordinates.
"""
import cv2
import numpy as np
from pathlib import Path

class FrameAnnotator:
    @staticmethod
    def annotate(frame: np.ndarray, label: str, confidence: float, bbox: tuple[int, int, int, int]) -> np.ndarray:
        """
        Draws a bounding rectangle around a matched face and places a labeled name banner over it.
        """
        # Create an explicit copy of the image layer to protect original frame memory states
        annotated_frame = frame.copy()
        x, y, w, h = bbox
        
        # Color profile definition: Bright, vibrant green outline box (BGR Format)
        box_color = (0, 255, 0)
        box_thickness = 2
        
        # Draw the main bounding container box around the face
        cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), box_color, box_thickness)
        
        # Structure text display string (e.g., "Anmol Gautam - 85.4%")
        text_content = f"{label.title()} - {confidence * 100:.1f}%"
        
        # Position calculations: Put text 10 pixels directly above the top edge of the box
        # max() ensures text stays visible on screen even if the face hits the top border
        text_position = (x, max(y - 10, 25))
        
        # Apply the label directly over the image array matrix
        cv2.putText(
            annotated_frame, 
            text_content, 
            text_position, 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.6,               # Font scaling size multiplier
            box_color,         # Text matching color profile
            2,                 # Stroke thickness pixels count
            cv2.LINE_AA        # Anti-aliased line rendering mode for smooth text edges
        )
        return annotated_frame

    @staticmethod
    def save_frame(frame: np.ndarray, output_path: Path) -> None:
        """
        Saves target frame layer as a high-quality JPEG to disk space locations.
        """
        # Force strict JPEG Quality setting to 95 to prevent visual compression artifacting
        cv2.imwrite(str(output_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])