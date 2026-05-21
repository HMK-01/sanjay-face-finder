from insightface.app import FaceAnalysis
import os
import urllib.request
import zipfile
import shutil  # 🎯 Added to securely handle nested folder extraction layouts

# 🎯 Establish safe directories inside the Streamlit cloud container
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
ANTELOPE_DIR = os.path.join(MODELS_DIR, 'antelopev2')

# 🌐 Official InsightFace pre-trained weights link
MODEL_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/antelopev2.zip"

def download_models_if_missing():
    """Guarantees neural net binaries are present inside the cloud container and properly flattened"""
    detection_file = os.path.join(ANTELOPE_DIR, 'scrfd_10g_bnkps.onnx')
    
    if not os.path.exists(detection_file):
        print("🚀 Cloud environment initialization: Downloading core AI models (~50MB compressed)...")
        os.makedirs(MODELS_DIR, exist_ok=True)
        
        zip_path = os.path.join(MODELS_DIR, "antelopev2.zip")
        
        # Stream the weights directly from the official mirror setup
        urllib.request.urlretrieve(MODEL_URL, zip_path)
        
        # Unpack the neural shapes right into the running workspace
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(MODELS_DIR)
            
        # 🎯 FIX THE NESTED FOLDER TRAP:
        # If it extracted into models/antelopev2/antelopev2/, move files up to models/antelopev2/
        nested_dir = os.path.join(ANTELOPE_DIR, 'antelopev2')
        if os.path.exists(nested_dir):
            for file_name in os.listdir(nested_dir):
                shutil.move(os.path.join(nested_dir, file_name), ANTELOPE_DIR)
            shutil.rmtree(nested_dir)
            
        # Clean up temporary archive file
        if os.path.exists(zip_path):
            os.remove(zip_path)
        print("✅ Models compiled, flattened, and active!")

# Execute the download verification check before launching the analyser
download_models_if_missing()

# Initialize the face analyzer pointing securely to our runtime directory
app = FaceAnalysis(name='antelopev2', root=MODELS_DIR, allowed_modules=['detection', 'recognition'])
app.prepare(ctx_id=-1, det_size=(640, 640)) # Forced CPU evaluation mode