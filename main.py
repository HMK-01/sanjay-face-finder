"""
Sanjay System Master Engine
Orchestrates face registration, matching pipelines, and CLI commands.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import cv2

from pipeline.detector import FaceDetector
from pipeline.embedder import FaceEmbedder
from pipeline.matcher import EmbeddingMatcher
from pipeline.annotator import FrameAnnotator
from pipeline.video_processor import VideoProcessor
import config

# In-memory vector database matching lowercase names to lists of 512-D embeddings
_EMBEDDING_REGISTRY: dict[str, list[np.ndarray]] = {}

# Initialize shared components once to save memory overhead
_detector = FaceDetector()
_embedder = FaceEmbedder()

def reset_registry() -> None:
    """Clears our memory database registry entirely."""
    global _EMBEDDING_REGISTRY
    _EMBEDDING_REGISTRY.clear()

def register_from_dataset(dataset_dir_path: str) -> None:
    """
    Scans a directory (like known_faces), reads subfolders as names,
    and automatically registers their face images into the AI memory.
    Gracefully skips problematic images without breaking the execution flow.
    """
    global _EMBEDDING_REGISTRY
    base_path = Path(dataset_dir_path)
    
    if not base_path.exists():
        print(f"⚠️ [System Alert] Directory '{dataset_dir_path}' not found in repo path. Auto-generating baseline tracking directory structure.")
        base_path.mkdir(parents=True, exist_ok=True)
        
    print("\n[AI Engine] Scanning dataset for known faces...")
    
    # Loop through every subfolder inside our dataset root
    for subfolder in base_path.iterdir():
        if subfolder.is_dir():
            person_name = subfolder.name
            
            # Find common image formats
            valid_images = []
            for img_path in subfolder.glob("*"):
                if img_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
                    valid_images.append(img_path)
            
            if not valid_images:
                print(f"⚠️ [Warning] Skipping folder '{person_name}' - No images found.")
                continue
                
            # Process photos one by one with a safety wrapper
            for img_path in valid_images:
                try:
                    register_persons({person_name: [str(img_path)]})
                except Exception as img_error:
                    print(f"⚠️ [Skipped File] '{img_path.name}' could not be registered: {img_error}")
                    continue

    # Final validation check
    _EMBEDDING_REGISTRY = {k: v for k, v in _EMBEDDING_REGISTRY.items() if v}
    if not _EMBEDDING_REGISTRY:
        raise ValueError("AI Registry is empty. None of the images provided in your dataset passed face validation rules.")

def register_persons(reference_dict: dict[str, list[str]]) -> None:
    """
    Processes reference images, ensures they contain a valid face,
    and updates the active embedding registry.
    """
    global _EMBEDDING_REGISTRY
    
    for person_name, image_paths in reference_dict.items():
        search_key = person_name.lower().strip()
        if search_key not in _EMBEDDING_REGISTRY:
            _EMBEDDING_REGISTRY[search_key] = []
            
        for path_str in image_paths:
            path = Path(path_str)
            if not path.exists():
                raise FileNotFoundError(f"Reference image missing at: {path}")

            # Read the image securely using UTF-8 friendly paths
            img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f"Could not decode or open image file: {path.name}")

            # Generate vectors directly using the full image context
            embedding = _embedder.extract_embedding(img)
            _EMBEDDING_REGISTRY[search_key].append(embedding)
            print(f"✓ Registered face model for '{person_name}' from {path.name}")

def search_all_persons_in_video(video_path: str) -> dict[str, dict]:
    """
    Scans a video track and updates visibility manifests for all registered identities.
    """
    video_path_obj = Path(video_path)
    processor = VideoProcessor(video_path_obj)
    
    # Pre-build our tracking manifests
    results_manifest = {}
    for name in _EMBEDDING_REGISTRY.keys():
        results_manifest[name] = {
            "status": "not_found",
            "person": name,
            "matches": [],
            "total_appearances": 0,
            "first_appearance_sec": None,
            "last_appearance_sec": None
        }

    def process_callback(frame: np.ndarray, frame_num: int, timestamp: float):
        detections = _detector.detect_faces(frame)
        if not detections:
            return

        h, w, _ = frame.shape

        for face in detections:
            x, y, width, height = face["box"]
            
            # 🎯 THE TRICK: Expand the video face slice out by 30% 
            # This gives ArcFace the context it needs to score matches above 0.65!
            pad_w = int(width * 0.3)
            pad_h = int(height * 0.3)
            
            x_min = max(0, x - pad_w)
            y_min = max(0, y - pad_h)
            x_max = min(w, x + width + pad_w)
            y_max = min(h, y + height + pad_h)
            
            face_crop = frame[y_min:y_max, x_min:x_max]
            if face_crop.size == 0:
                continue
            
            try:
                current_embedding = _embedder.extract_embedding(face_crop)
                matched_name, confidence = EmbeddingMatcher.find_best_match(current_embedding, _EMBEDDING_REGISTRY)
                
                if matched_name != "Unknown":
                    target_key = matched_name.lower().strip()
                    manifest = results_manifest[target_key]
                    manifest["status"] = "found"
                    
                    out_img_name = f"{target_key}_frame_{frame_num}.jpg"
                    out_img_path = config.OUTPUT_DIR / out_img_name
                    
                    # Annotate original face box cleanly
                    annotated_img = FrameAnnotator.annotate(frame, matched_name, confidence, face["box"])
                    FrameAnnotator.save_frame(annotated_img, out_img_path)

                    match_record = {
                        "frame_number": frame_num,
                        "timestamp_seconds": round(timestamp, 2),
                        "confidence": round(confidence, 4),
                        "bounding_box": {"x": x, "y": y, "width": width, "height": height},
                        "frame_image_path": str(out_img_path)
                    }
                    manifest["matches"].append(match_record)
                    manifest["total_appearances"] += 1
                    
                    if manifest["first_appearance_sec"] is None:
                        manifest["first_appearance_sec"] = round(timestamp, 2)
                    manifest["last_appearance_sec"] = round(timestamp, 2)

            except Exception:
                continue
            
    # Fire off frame-streaming logic
    processor.process_frames(process_callback)
    return results_manifest

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sanjay AI Person Re-Identification Pipeline CLI")
    parser.add_argument("--video", type=str, required=True, help="Path to your input video file")
    parser.add_argument("--dataset", type=str, default="known_faces", help="Path to your known faces dataset directory")
    parser.add_argument("--threshold", type=float, default=0.50, help="Confidence threshold match limit")
    parser.add_argument("--sample-rate", type=int, default=5, help="Process every Nth frame")

    args = parser.parse_args()

    # Dynamic configurations updates override
    config.SIMILARITY_THRESHOLD = args.threshold
    config.FRAME_SAMPLE_RATE = args.sample_rate

    # Run system loop pipelines
    reset_registry()
    try:
        register_from_dataset(args.dataset)
        all_results = search_all_persons_in_video(args.video)
        
        print("\n================ COMPLETE VIDEO SWEEP REPORT ================")
        print(json.dumps(all_results, indent=2))
        print("=======================================================")
    except Exception as e:
        print(f"\n❌ Execution Error Encountered: {e}")