from insightface.app import FaceAnalysis
import os
import urllib.request
import zipfile

# 🎯 Establish safe directories inside the Streamlit cloud container
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
ANTELOPE_DIR = os.path.join(MODELS_DIR, 'antelopev2')

# 🌐 Official InsightFace pre-trained weights link
MODEL_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/antelopev2.zip"

def download_models_if_missing():
    """Guarantees neural net binaries are present inside the cloud container before processing"""
    if not os.path.exists(ANTELOPE_DIR):
        print("🚀 Cloud environment initialization: Downloading core AI models (~50MB compressed)...")
        os.makedirs(MODELS_DIR, exist_ok=True)
        
        zip_path = os.path.join(MODELS_DIR, "antelopev2.zip")
        
        # Stream the weights directly from the official mirror setup
        urllib.request.urlretrieve(MODEL_URL, zip_path)
        
        # Unpack the neural shapes right into the running workspace
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(MODELS_DIR)
            
        # Clean up temporary archive file
        os.remove(zip_path)
        print("✅ Models compiled and active!")

# Execute the download verification check before launching the analyser
download_models_if_missing()

# Initialize the face analyzer pointing securely to our runtime directory
app = FaceAnalysis(name='antelopev2', root=MODELS_DIR, allowed_modules=['detection', 'recognition'])
app.prepare(ctx_id=-1, det_size=(640, 640)) # Forced CPU evaluation mode