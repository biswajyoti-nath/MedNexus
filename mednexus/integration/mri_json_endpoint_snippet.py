# Add these routes to the existing NeuroVision app.py.
# Ensure `jsonify` is imported from Flask.

@app.route('/health', methods=['GET'])
def mednexus_health():
    return jsonify({
        'status': 'ok',
        'specialist': 'NeuroVision MRI',
        'task': '2D brain MRI slice screening',
        'clinically_validated': False,
    })


@app.route('/api/predict', methods=['POST'])
def api_predict():
    if 'file' not in request.files:
        return jsonify({'error': 'Missing file field'}), 400

    file = request.files['file']
    if not file.filename or not allowed_file(file.filename):
        return jsonify({'error': 'A .tif or .tiff MRI slice is required'}), 400

    import base64
    import io
    import uuid

    import numpy as np
    from PIL import Image

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{uuid.uuid4().hex}_{filename}")
    file.save(filepath)

    try:
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
        try:
            os.remove(filepath)
        except OSError:
            pass
