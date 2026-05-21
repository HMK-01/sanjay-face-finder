from insightface.app import FaceAnalysis
import os
import urllib.request
import zipfile
import shutil

# 🎯 Universal safe directory routing inside the cloud workspace
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
ANTELOPE_DIR = os.path.join(MODELS_DIR, 'antelopev2')

MODEL_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/antelopev2.zip"

def download_and_flatten_models():
    """Ensures AI weight binaries are present and perfectly structured without nested folder traps"""
    target_onnx = os.path.join(ANTELOPE_DIR, 'scrfd_10g_bnkps.onnx')
    
    if not os.path.exists(target_onnx):
        os.makedirs(MODELS_DIR, exist_ok=True)
        zip_path = os.path.join(MODELS_DIR, "antelopev2.zip")
        
        print("🚀 Downloading core AI model assets (~50MB)...")
        urllib.request.urlretrieve(MODEL_URL, zip_path)
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(MODELS_DIR)
            
        # 📂 Fix the nested folder trap if models extracted to models/antelopev2/antelopev2/
        nested_dir = os.path.join(ANTELOPE_DIR, 'antelopev2')
        if os.path.exists(nested_dir):
            for file_name in os.listdir(nested_dir):
                shutil.move(os.path.join(nested_dir, file_name), ANTELOPE_DIR)
            shutil.rmtree(nested_dir)
            
        if os.path.exists(zip_path):
            os.remove(zip_path)
        print("✅ Models structural layout verified!")

# Run download check before initialization
download_and_flatten_models()

# Initialize single global inference instance pointing directly to our directory
app = FaceAnalysis(name='antelopev2', root=MODELS_DIR, allowed_modules=['detection', 'recognition'])
app.prepare(ctx_id=-1, det_size=(640, 640)) # Balanced CPU execution mode