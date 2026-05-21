from insightface.app import FaceAnalysis
import os
import urllib.request
import zipfile
import shutil

# 🎯 Universal safe directory routing inside the cloud workspace
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
BUFFALO_DIR = os.path.join(MODELS_DIR, 'buffalo_l')

# 🌐 Official InsightFace pre-trained Buffalo_L weights repository link
MODEL_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"

def download_and_flatten_models():
    """Forces clean extraction of Buffalo_L ONNX files directly into models/buffalo_l/"""
    # Look for the primary face detection model inside buffalo_l
    target_onnx = os.path.join(BUFFALO_DIR, 'det_500m.onnx')
    
    if not os.path.exists(target_onnx):
        print("🚀 Core Buffalo_L models missing. Initiating download channel (~145MB)...")
        shutil.rmtree(BUFFALO_DIR, ignore_errors=True)
        os.makedirs(BUFFALO_DIR, exist_ok=True)
        
        zip_path = os.path.join(MODELS_DIR, "buffalo_l.zip")
        temp_extract_dir = os.path.join(MODELS_DIR, "temp_extract")
        
        # Download official Buffalo_L archive bundle
        urllib.request.urlretrieve(MODEL_URL, zip_path)
        
        # Unpack to temporary staging folder
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_extract_dir)
            
        # 🎯 Move all .onnx model layers directly to models/buffalo_l/ without nested folders
        for root, _, files in os.walk(temp_extract_dir):
            for file in files:
                if file.endswith('.onnx'):
                    shutil.move(os.path.join(root, file), os.path.join(BUFFALO_DIR, file))
                    
        # 🧼 Clean up workspace archives
        shutil.rmtree(temp_extract_dir, ignore_errors=True)
        if os.path.exists(zip_path):
            os.remove(zip_path)
            
        print("✅ Buffalo_L models compiled and ready!")

# Run structural checks before initializing
download_and_flatten_models()

# Initialize the shared engine instance pointing strictly to Buffalo_L
app = FaceAnalysis(name='buffalo_l', root=MODELS_DIR, allowed_modules=['detection', 'recognition'])
app.prepare(ctx_id=-1, det_size=(640, 640))  # Force CPU execution to comply with Streamlit limits