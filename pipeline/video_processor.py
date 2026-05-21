"""
Video Streaming Engine
Handles frame-by-frame extraction without overloading system memory.
"""
import cv2
from pathlib import Path
from typing import Callable
from config import FRAME_SAMPLE_RATE

class VideoProcessor:
    def __init__(self, video_path: Path):
        self.video_path = video_path
        if not self.video_path.exists():
            raise FileNotFoundError(f"Target video file not found at: {self.video_path}")

    def process_frames(self, callback: Callable[[cv2.Mat, int, float], None]) -> None:
        """
        Streams video frame-by-frame. Applies an execution callback 
        function on targeted sampled frame indexes.
        """
        # Open a read pointer stream handle to the raw video file
        cap = cv2.VideoCapture(str(self.video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Guard rail: Fallback if the file metadata doesn't report a valid frame rate
        if fps == 0:
            fps = 30.0

        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break # Reached the end of the video file or hit a bad frame packet

            # Performance Tuning: Only analyze frames matching our configured step skip rule
            if frame_idx % FRAME_SAMPLE_RATE == 0:
                # Calculate exact position time metric within the video timeline
                timestamp_sec = float(frame_idx / fps)
                try:
                    # Run the processing logic on our frame
                    callback(frame, frame_idx, timestamp_sec)
                except Exception as error:
                    print(f"[Warning] Skipping frame index {frame_idx} due to an error: {error}")
            
            frame_idx += 1

        # Close the file stream pointer cleanly
        cap.release()