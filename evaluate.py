"""
Sanjay System Model Performance Evaluation Suite
Computes Precision, Recall, and F1-Score metrics against ground truth.
"""
import json
from pathlib import Path

def calculate_metrics(ground_truth: dict[int, str], ai_results: dict) -> None:
    """
    Compares AI outputs against human-verified ground truth to compute 
    Precision, Recall, and F1-Score accuracy metrics.
    """
    print("\n================ SYSTEM ACCURACY EVALUATION ================")
    
    # Extract all frames where the AI claimed to find the specific person
    target_person = "prasang charitra"  # Our target evaluation identity
    ai_matched_frames = set()
    
    if target_person in ai_results:
        for match in ai_results[target_person]["matches"]:
            ai_matched_frames.add(match["frame_number"])
            
    # Extract all frames where the human verified the person actually IS there
    true_positive_frames = set()
    for frame_num, actual_name in ground_truth.items():
        if actual_name.lower().strip() == target_person:
            true_positive_frames.add(frame_num)

    # Calculate Core Confusion Matrix Variables
    # 1. True Positives (TP): AI said they were there, and they actually were.
    tp = len(ai_matched_frames.intersection(true_positive_frames))
    
    # 2. False Positives (FP): AI said they were there, but it was a stranger (Wrong Person marked).
    fp = len(ai_matched_frames - true_positive_frames)
    
    # 3. False Negatives (FN): Person was actually there, but the AI missed them completely.
    fn = len(true_positive_frames - ai_matched_frames)

    # Compute Statistical Accuracy Formulas
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    # Output Performance Dashboard
    print(f"📊 Evaluation Target Profile: '{target_person}'")
    print("------------------------------------------------------------")
    print(f"✅ True Positives (Correctly Identified) : {tp} frames")
    print(f"❌ False Positives (Wrong Person Marked) : {fp} frames")
    print(f"📉 False Negatives (Missed Appearances)  : {fn} frames")
    print("------------------------------------------------------------")
    print(f"🎯 Precision (Exactness Quality)         : {round(precision * 100, 2)}%")
    print(f"📢 Recall (Detection Completeness)       : {round(recall * 100, 2)}%")
    print(f"🏆 TOTAL SYSTEM F1-SCORE                 : {round(f1_score * 100, 2)}%")
    print("=============================================================")

if __name__ == "__main__":
    # 📝 1. MANUAL GROUND TRUTH SETUP (The Human Verification Map)
    # Map the frame number to who is ACTUALLY sitting/standing in that frame.
    # Let's populate this based on the video frames we verified earlier!
    ground_truth_map = {
        15: "prasang charitra",
        35: "prasang charitra",
        155: "prasang charitra",
        200: "prasang charitra",
        240: "prasang charitra",  # Remember frame 240 was correct!
        440: "anmol gautam",      # Frame 440 was the stranger/Anmol frame!
        490: "prasang charitra",
        495: "prasang charitra",
        500: "prasang charitra",
        505: "prasang charitra",
        525: "prasang charitra"
    }

    # 📝 2. LOAD LATEST GENERATED RUN REPORT
    # First, let's make sure you run a command-line search to generate a fresh report data file.
    # If main.py doesn't automatically drop a copy, let's simulate reading it directly here.
    # For testing, we can write out your specific previous run values.
    
    # Let's mock a data load representing the run you just executed to test the logic:
    sample_ai_report = {
      "prasang charitra": {
        "matches": [
          {"frame_number": 15}, {"frame_number": 35}, {"frame_number": 155}, 
          {"frame_number": 200}, {"frame_number": 240}, {"frame_number": 440}, # <- AI caught the stranger here
          {"frame_number": 490}, {"frame_number": 495}, {"frame_number": 500}, 
          {"frame_number": 505}, {"frame_number": 525}
        ]
      }
    }
    
    calculate_metrics(ground_truth_map, sample_ai_report)