"""
Sanjay System Consumer Face Search Web App
Clean, jargon-free user interface optimized for production deployment.
"""
import sys
sys.modules['cv2'] = __import__('cv2')
import streamlit as st
import cv2
import numpy as np
import tempfile
from pathlib import Path

# --- PAGE LAYOUT CONFIGURATIONS ---
st.set_page_config(
    page_title="Sanjay - AI Based Re-identification Model",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed" # Hide sidebar entirely for a clean main stage interface
)

# Custom minimal CSS styling to polish component padding
st.markdown("""
    <style>
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    .stAlert p { font-size: 1rem; margin-bottom: 0px !important; }
    </style>
""", unsafe_allow_html=True)

# --- APP HEADER ---
st.title("🎯 Sanjay - AI Based Re-identification Model")
st.markdown("Upload a photo of a person to scan, track, and extract every moment they appear inside a video clip.")

# --- ONBOARDING INSTRUCTIONS FOR FIRST-TIME USERS ---
with st.expander("ℹ️ First Time User? Click here for 3 simple steps to get started", expanded=True):
    st.markdown("""
    Welcome! This app uses smart face recognition to scan videos and locate a specific person automatically. 
    
    1. **Upload a clear photo** of the single person you want to find under the **Person's Photo** zone.
    2. **Upload your video file** into the **Video File** zone.
    3. **Adjust settings if needed:** 
       * *Match Accuracy (Strictness)*: Higher numbers mean less room for error (prevents matching strangers).
       * *Processing Speed*: Bypasses frames to scan longer videos significantly faster.
    4. Click the **Start Search Engine** button and watch the results populate!
    """)

# --- LOAD UNDERLYING ENGINES ---
@st.cache_resource
def load_vision_pipelines():
    from pipeline.detector import FaceDetector
    from pipeline.embedder import FaceEmbedder
    return FaceDetector(), FaceEmbedder()

try:
    detector, embedder = load_vision_pipelines()
except Exception as init_err:
    st.error(f"Failed to initialize core video processing layers: {init_err}")
    st.stop()

st.markdown("---")

# --- MAIN DRAG-AND-DROP FILE INTERFACES (FRONT STAGE) ---
col_face, col_video = st.columns(2)

with col_face:
    st.subheader("🧑 1. Person's Photo")
    uploaded_face = st.file_uploader(
        "Upload a clear portrait picture containing the target face", 
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

with col_video:
    st.subheader("🎬 2. Video File")
    uploaded_video = st.file_uploader(
        "Upload the video clip track you want to search through", 
        type=["mp4", "mov", "avi"],
        label_visibility="collapsed"
    )

st.markdown("---")

# --- USER-FRIENDLY ADJUSTMENT SLIDERS ---
st.subheader("🎛️ Fine-Tune Search Engine Settings")
col_slider1, col_slider2 = st.columns(2)

with col_slider1:
    # Re-mapped from raw Cosine Similarity Threshold to an intuitive "Strictness Level" percentage scale
    strictness_ui = st.slider(
        "Match Accuracy Level (Strictness)", 
        min_value=40, max_value=95, value=65, step=1,
        help="Higher percentages reduce mistakes, ensuring the system only saves frames where it is completely confident it found the right person."
    )
    threshold = float(strictness_ui / 100.0)

with col_slider2:
    # Re-mapped from Sample Rate to an intuitive "Processing Mode" speed selection mapping
    speed_mode = st.select_slider(
        "Video Scanning Speed",
        options=["Thorough (Slowest)", "Standard (Balanced)", "Express (Fast)", "Turbo (Fastest)"],
        value="Standard (Balanced)",
        help="Thorough checks every frame. Fast modes skip redundant frames to deliver your summary report in seconds."
    )
    
    speed_map = {
        "Thorough (Slowest)": 1,
        "Standard (Balanced)": 5,
        "Express (Fast)": 12,
        "Turbo (Fastest)": 24
    }
    sample_rate = speed_map[speed_mode]

st.markdown("<br>", unsafe_allow_html=True)

# --- PROCESS EXECUTION BLOCK ---
if st.button("🔍 Start Search Engine", type="primary", use_container_width=True):
    if not uploaded_face or not uploaded_video:
        st.error("⚠️ Please make sure both a **Person's Photo** and a **Video File** are uploaded before starting.")
    else:
        st.toast("Preparing video analysis components...", icon="⏳")
        
        # Parse face image from computer memory stream
        face_bytes = np.frombuffer(uploaded_face.read(), np.uint8)
        ref_image = cv2.imdecode(face_bytes, cv2.IMREAD_COLOR)
        
        try:
            target_embedding = embedder.extract_embedding(ref_image)
        except Exception as face_err:
            st.error(f"❌ Could not register target face picture profile: Ensure a human face is clearly visible and well-lit.")
            st.stop()
            
        # Secure video stream data into a temporary storage file path safely
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video_file:
            temp_video_file.write(uploaded_video.read())
            temp_video_path = temp_video_file.name

        # Read input metrics details using OpenCV file trackers
        video_capture = cv2.VideoCapture(temp_video_path)
        total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = video_capture.get(cv2.CAP_PROP_FPS)
        
        if total_frames <= 0:
            st.error("Unable to properly evaluate files structure formats. Double-check your media video track.")
            st.stop()

        # Instantiate live progress indicator bars on user display dashboard lines
        progress_bar = st.progress(0.0, text="Initializing video streaming layers...")
        match_records_list = []
        frame_counter = 0
        
        # Process clip array streams
        while video_capture.isOpened():
            ret, frame = video_capture.read()
            if not ret:
                break
                
            frame_counter += 1
            if frame_counter % sample_rate != 0:
                continue
                
            progress_percent = min(frame_counter / total_frames, 1.0)
            progress_bar.progress(progress_percent, text=f"Scanning Video File: Frame {frame_counter} of {total_frames}")
            
            detections = detector.detect_faces(frame)
            if not detections:
                continue
                
            h_frame, w_frame, _ = frame.shape
            
            for face in detections:
                x, y, width, height = face["box"]
                
                # Apply 30% background perspective context border rules to crops matrices calculations
                pad_w, pad_h = int(width * 0.3), int(height * 0.3)
                x_min, y_min = max(0, x - pad_w), max(0, y - pad_h)
                x_max, y_max = min(w_frame, x + width + pad_w), min(h_frame, y + height + pad_h)
                
                face_crop = frame[y_min:y_max, x_min:x_max]
                if face_crop.size == 0:
                    continue
                    
                try:
                    current_embedding = embedder.extract_embedding(face_crop)
                    
                    dot_p = np.dot(target_embedding, current_embedding)
                    norm_r = np.linalg.norm(target_embedding) * np.linalg.norm(current_embedding)
                    similarity_score = float(dot_p / norm_r) if norm_r != 0 else 0.0
                    
                    if similarity_score >= threshold:
                        timestamp_sec = round(frame_counter / fps, 2)
                        
                        # Generate annotated canvas image boxes targets labels
                        annotated_canvas = frame.copy()
                        cv2.rectangle(annotated_canvas, (x, y), (x + width, y + height), (0, 255, 0), 3)
                        label_string = f"MATCH: {round(similarity_score * 100, 1)}%"
                        cv2.putText(annotated_canvas, label_string, (x, max(30, y - 12)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        
                        rgb_frame = cv2.cvtColor(annotated_canvas, cv2.COLOR_BGR2RGB)
                        
                        match_records_list.append({
                            "frame_idx": frame_counter,
                            "time_offset": timestamp_sec,
                            "score": similarity_score,
                            "img_array": rgb_frame
                        })
                        
                except Exception:
                    continue

        # Release active hardware system thread handles securely from file memory caches registers
        video_capture.release()
        Path(temp_video_path).unlink(missing_ok=True)
        progress_bar.empty()
        st.balloons()
        
        # --- RENDER RESULTS PANEL GALLERY GRID ---
        st.subheader(f"📊 Results Summary (Found {len(match_records_list)} Match Instances)")
        
        if len(match_records_list) == 0:
            st.info("ℹ️ No matching face instances found. Try sliding down the 'Match Accuracy' setting if the person is moving fast or poorly lit.")
        else:
            columns_layout_grid = st.columns(3)
            for index, match in enumerate(match_records_list):
                col_target = columns_layout_grid[index % 3]
                with col_target:
                    card_title = f"⏱️ Spotted at: {match['time_offset']} seconds"
                    col_target.image(match["img_array"], caption=card_title, use_container_width=True)
                    st.markdown(f"**Confidence Level:** `{round(match['score'] * 100, 1)}% Match`")
                    st.markdown("---")