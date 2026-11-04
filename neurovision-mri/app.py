import os
import sys
import uuid
import io
import base64
import shutil
import threading
import numpy as np
from PIL import Image
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request, url_for, jsonify
import torch
import matplotlib
matplotlib.use('Agg')

# Ensure src module is accessible
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
sys.path.append(os.path.dirname(__file__))

# MRI inference, including automatic XAI passes, must stay on the GPU.
mri_device_env = os.getenv('MRI_DEVICE', 'cuda').strip().lower()
if mri_device_env != 'cuda':
    raise RuntimeError("NeuroVision is GPU-only in this demo. Set MRI_DEVICE=cuda; CPU inference is disabled.")
if not torch.cuda.is_available():
    raise RuntimeError(
        "MRI_DEVICE=cuda was requested, but this PyTorch build or system cannot access CUDA. "
        "Refusing to fall back to CPU inference. Install a CUDA-enabled PyTorch build and verify the NVIDIA driver."
    )
DEVICE = torch.device('cuda')

from src.inference import load_ensembles, run_inference
from src.explainability.compare import generate_full_explanation

app = Flask(__name__)
MODEL_LOCK = threading.Lock()

# Configure upload and XAI folders
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
XAI_FOLDER = os.path.join(BASE_DIR, 'static', 'xai')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(XAI_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['XAI_FOLDER'] = XAI_FOLDER

# Allowed extensions (.tif and .tiff)
ALLOWED_EXTENSIONS = {'tif', 'tiff'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

print(f"Initializing Flask App and Loading Models on {DEVICE} (This may take a moment)...")
clf_ens, seg_ens = load_ensembles(DEVICE)
print("Models loaded successfully!")

@app.route('/health', methods=['GET'])
@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'specialist': 'NeuroVision MRI',
        'task': '2D brain MRI slice screening',
        'device': DEVICE.type,
        'gpu_name': torch.cuda.get_device_name(DEVICE) if DEVICE.type == 'cuda' else None,
        'clinically_validated': False
    })

@app.route('/api/predict', methods=['POST'])
def api_predict():
    if 'file' not in request.files:
        return jsonify({'error': 'Missing file field'}), 400

    file = request.files['file']
    if not file.filename or not allowed_file(file.filename):
        return jsonify({'error': 'A .tif or .tiff MRI slice is required'}), 400

    filename = secure_filename(file.filename)
    if not filename:
        filename = "upload.tif"
    unique_name = f"{uuid.uuid4().hex}_{filename}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_name)
    file.save(filepath)

    try:
        with MODEL_LOCK:
            prob, has_tumor, mask_np, bbox, orig_img = run_inference(
                filepath, clf_ens, seg_ens, DEVICE
            )
        overlay_data_url = None
        if mask_np is not None:
            source = Image.fromarray(orig_img).convert('RGBA')
            mask = Image.fromarray((mask_np > 0).astype(np.uint8) * 255)
            mask = mask.resize(source.size, Image.Resampling.NEAREST)
            tint = Image.new('RGBA', source.size, (61, 190, 139, 0))
            tint.putalpha(mask.point(lambda value: 112 if value else 0))
            overlay = Image.alpha_composite(source, tint)
            overlay.thumbnail((760, 760))
            buffer = io.BytesIO()
            overlay.save(buffer, format='PNG', optimize=True)
            overlay_data_url = 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode('ascii')

        return jsonify({
            'specialist': 'NeuroVision MRI',
            'task': '2D brain MRI slice screening',
            'predicted_class': 'tumour_suspected' if has_tumor else 'no_tumour_detected_by_model',
            'model_score': float(prob),
            'score_type': 'model output; not a calibrated clinical probability',
            'segmentation_available': mask_np is not None,
            'bbox': [int(x) for x in bbox] if bbox is not None else None,
            'segmentation_overlay': overlay_data_url,
            'image_width': int(orig_img.shape[1]),
            'image_height': int(orig_img.shape[0]),
            'scope': 'Research model trained on LGG MRI slices; not validated for broad neurological diagnosis or patient-level diagnosis.',
            'clinically_validated': False
        })
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500
    finally:
        if filepath and os.path.exists(filepath):
            try:
                os.remove(filepath)
            except OSError:
                pass


@app.route('/api/explain', methods=['POST'])
def api_explain():
    """Return temporary, slice-level model visualizations without changing /api/predict."""
    if 'file' not in request.files:
        return jsonify({'error': 'Missing file field'}), 400

    file = request.files['file']
    if not file.filename or not allowed_file(file.filename):
        return jsonify({'error': 'A .tif or .tiff MRI slice is required'}), 400

    filename = secure_filename(file.filename) or 'scan.tif'
    unique_name = f"{uuid.uuid4().hex}_{filename}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_name)
    output_dir = os.path.join(app.config['XAI_FOLDER'], uuid.uuid4().hex)
    os.makedirs(output_dir, exist_ok=False)
    file.save(filepath)

    try:
        # Reuse the already-loaded ensembles and serialize this expensive optional pass.
        with MODEL_LOCK:
            generate_full_explanation(filepath, clf_ens, seg_ens, DEVICE, output_dir)

        stem = unique_name.replace('.tif', '')

        def read_data_url(path):
            if not os.path.isfile(path):
                return None
            with open(path, 'rb') as image_file:
                encoded = base64.b64encode(image_file.read()).decode('ascii')
            return 'data:image/png;base64,' + encoded

        with Image.open(filepath) as source_image:
            original = source_image.convert('RGB')
            original.thumbnail((760, 760))
            buffer = io.BytesIO()
            original.save(buffer, format='PNG', optimize=True)
            original_data_url = 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode('ascii')

        return jsonify({
            'original': original_data_url,
            'maps': {
                'gradcam': read_data_url(os.path.join(output_dir, f'{stem}_gradcam.png')),
                'agreement': read_data_url(os.path.join(output_dir, f'{stem}_agreement.png')),
                'uncertainty': read_data_url(os.path.join(output_dir, f'{stem}_uncertainty.png')),
            },
            'scope': 'Exploratory model visualizations of one 2D research slice; not validated clinical explanations or diagnostic evidence.',
            'clinically_validated': False,
        })
    except Exception as exc:
        return jsonify({'error': f'Could not generate MRI model visualizations: {exc}'}), 500
    finally:
        for path in (filepath,):
            if os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass
        shutil.rmtree(output_dir, ignore_errors=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return render_template('index.html', error="No file part")
        
        file = request.files['file']
        
        if file.filename == '':
            return render_template('index.html', error="No selected file")
            
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            if not filename:
                filename = "scan.tif"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            try:
                # 1. Run Classification and Segmentation
                with MODEL_LOCK:
                    prob, has_tumor, mask_np, bbox, orig_img = run_inference(filepath, clf_ens, seg_ens, DEVICE)
                confidence = prob * 100
                
                base_name = filename.rsplit('.', 1)[0]
                
                # Convert the image to PNG for web display (browsers can't render .tif)
                display_filename = f"{base_name}.png"
                Image.fromarray(orig_img).save(os.path.join(app.config['UPLOAD_FOLDER'], display_filename))
                
                results = {
                    'tumor_present': has_tumor,
                    'confidence': confidence,
                    'prob': float(prob),
                    'img_original': f"uploads/{display_filename}"
                }
                
                # 2. Run Explainable AI if Tumor is Present
                if has_tumor:
                    with MODEL_LOCK:
                        generate_full_explanation(filepath, clf_ens, seg_ens, DEVICE, app.config['XAI_FOLDER'])
                    
                    results['img_gradcam'] = f"xai/{base_name}_gradcam.png"
                    results['img_agreement'] = f"xai/{base_name}_agreement.png"
                    results['img_uncertainty'] = f"xai/{base_name}_uncertainty.png"
                    
                return render_template('index.html', results=results)
                
            except Exception as e:
                return render_template('index.html', error=str(e))
                
        else:
            return render_template('index.html', error="Invalid file format. Please upload a .tif or .tiff file.")

    # GET request
    return render_template('index.html')

if __name__ == '__main__':
    port = int(os.getenv('PORT', 8080))
    app.run(host='127.0.0.1', port=port, debug=False)
