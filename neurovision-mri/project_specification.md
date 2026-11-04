# 1. Project Information
- Project Name: Brain Tumor MRI Segmentation & Explainable AI Pipeline
- Project Type: Deep Learning / Web Application / Medical Imaging
- Domain: Medical Artificial Intelligence / Neuro-oncology
- Purpose: Automated MRI classification and segmentation pipeline providing pixel-level accuracy and clinical transparency while operating strictly within consumer-grade 4GB VRAM constraint.
- Problem it solves: Manual MRI segmentation is slow and subjective. Clinical adoption of Deep Learning is hindered by black-box models and extreme hardware requirements (24GB+ GPUs).
- Target Users: Medical Professionals, Radiologists, Doctors
- Main Features: Tumor Detection (Binary Classification Ensemble), Tumor Boundary Delineation (Segmentation Ensemble), Explainable AI (Grad-CAM, Agreement Maps, Monte Carlo Dropout Uncertainty), Clinical Web Dashboard.
- Expected Inputs: MRI Scan Images (.tif, .png, .jpg, .jpeg)
- Expected Outputs: Binary Classification (Tumor/No Tumor), Detection Confidence Score, Tumor Bounding Box, Coordinate Mask, Clinical Heatmaps (.png)

---------------------------------------------------------

# 2. Technology Stack
- Programming Languages: Python 3.12, HTML5, CSS3
- Frameworks: PyTorch, Flask
- Libraries: Werkzeug, Pillow (PIL), OpenCV (opencv-python-headless), Albumentations
- ML Libraries: scikit-learn
- DL Libraries: torch, torchvision, timm, segmentation_models_pytorch
- Visualization Libraries: matplotlib, seaborn
- Database: None
- Backend: Flask
- Frontend: HTML/CSS (Jinja2 Templates)
- Deployment: Local / Unknown external cloud deployment
- Cloud Services: Unknown
- Package Managers: pip (requirements.txt)
- Build Tools: None
- Operating System Support: Windows / Linux / macOS

---------------------------------------------------------

# 3. Repository Structure
```
neurovision-mri/
│
├── checkpoints/                    # Saved weights for trained classification and segmentation models, and training logs
├── kaggle_3m/                      # Kaggle LGG MRI Segmentation dataset (Patient folders + CSV)
├── results/                        # Generated output metrics, charts, confusion matrices, and XAI heatmaps
├── sample_images/                  # Sample MRI images for quick testing
├── src/                            # Core Deep Learning logic and utility scripts
│   ├── explainability/             # Explainable AI scripts (Grad-CAM, Agreement Maps, Uncertainty)
│   ├── models/                     # Deep learning architectures (Classifiers and Segmentors)
├── static/                         # Frontend web assets (CSS) and dynamically generated uploads/XAI images
│   ├── css/                        
│   ├── uploads/                    
│   ├── xai/                        
├── templates/                      # Flask HTML Jinja2 templates (index.html)
├── venv/                           # Python Virtual Environment
├── environments/                   # Python Virtual Environment (referenced in GUIDE.md)
```

---------------------------------------------------------

# 4. Complete File Inventory

- Filename: `app.py`
  - Extension: `.py`
  - Purpose: Main Flask web server for the clinical UI
  - Language: Python
  - Size: 3581 bytes
  - Dependencies: Flask, Werkzeug, PIL, src.config, src.inference, src.explainability.compare
  - Imported by: None
  - Imports: os, sys, werkzeug.utils.secure_filename, flask.Flask, flask.render_template, flask.request, flask.url_for, src.config.DEVICE, src.inference.load_ensembles, src.inference.run_inference, matplotlib, src.explainability.compare.generate_full_explanation, shutil, PIL.Image
  - Executed directly?: Yes
  - Main functions: `allowed_file`, `index`
  - Main classes: None
  - Output produced: Flask web server running on port 8080

- Filename: `config.py`
  - Extension: `.py`
  - Purpose: Project configuration, hyperparameters, and path constants
  - Language: Python
  - Size: 1015 bytes
  - Dependencies: torch
  - Imported by: `app.py`, `src/evaluate.py`, `src/inference.py`, `src/train_classifier.py`, `src/train_segmentor.py`, `src/utils.py`, `src/explainability/compare.py`
  - Imports: os, torch
  - Executed directly?: No
  - Main functions: None
  - Main classes: None
  - Output produced: Environment configurations

- Filename: `dataset.py`
  - Extension: `.py`
  - Purpose: PyTorch Dataset loader and Augmentation pipelines
  - Language: Python
  - Size: 2188 bytes
  - Dependencies: numpy, PIL, torch, albumentations
  - Imported by: `src/evaluate.py`, `src/inference.py`, `src/train_classifier.py`, `src/train_segmentor.py`
  - Imports: os, glob, numpy, PIL.Image, torch, torch.utils.data.Dataset, albumentations, albumentations.pytorch.ToTensorV2
  - Executed directly?: No
  - Main functions: `get_train_transform`, `get_val_transform`
  - Main classes: `BrainMRIDataset`
  - Output produced: Loaded tensors and labels for training/inference

- Filename: `ensemble.py`
  - Extension: `.py`
  - Purpose: Defines soft voting and mask averaging wrappers for models
  - Language: Python
  - Size: 1732 bytes
  - Dependencies: torch
  - Imported by: `src/inference.py`
  - Imports: torch, torch.nn
  - Executed directly?: No
  - Main functions: None
  - Main classes: `ClassifierEnsemble`, `SegmentorEnsemble`
  - Output produced: Ensembled predictions

- Filename: `evaluate.py`
  - Extension: `.py`
  - Purpose: Evaluates ensemble performance (Accuracy, Dice, IoU) on validation/test set
  - Language: Python
  - Size: 4324 bytes
  - Dependencies: argparse, numpy, torch, tqdm, sklearn, matplotlib, seaborn
  - Imported by: None
  - Imports: os, argparse, numpy, torch, torch.utils.data.DataLoader, tqdm.tqdm, sklearn.metrics, matplotlib.pyplot, seaborn, src.config, src.dataset, src.inference, src.utils
  - Executed directly?: Yes
  - Main functions: `evaluate_classification`, `evaluate_segmentation`, `plot_confusion_matrix`
  - Main classes: None
  - Output produced: Console metrics, `test_confusion_matrix.png` / `val_confusion_matrix.png`

- Filename: `inference.py`
  - Extension: `.py`
  - Purpose: Runs an individual MRI image through both ensembles to generate predictions
  - Language: Python
  - Size: 4456 bytes
  - Dependencies: argparse, numpy, torch, cv2, matplotlib, PIL
  - Imported by: `app.py`, `src/explainability/compare.py`
  - Imports: os, argparse, numpy, torch, cv2, matplotlib.pyplot, PIL.Image, sys, src.config, src.dataset.get_val_transform, src.models.classifier.BrainClassifier, src.models.segmentor.BrainSegmentor, src.ensemble, src.utils.load_checkpoint
  - Executed directly?: Yes
  - Main functions: `extract_bounding_box`, `run_inference`, `load_ensembles`, `visualize_prediction`
  - Main classes: None
  - Output produced: Annotated plots of predictions

- Filename: `train_classifier.py`
  - Extension: `.py`
  - Purpose: Training script for the binary classifiers (ResNet, EfficientNet, DenseNet)
  - Language: Python
  - Size: 5814 bytes
  - Dependencies: argparse, json, torch, tqdm, sklearn
  - Imported by: None
  - Imports: os, argparse, json, torch, torch.nn, torch.utils.data.DataLoader, tqdm.tqdm, sklearn.metrics, sys, src.config, src.utils, src.dataset, src.models.classifier.BrainClassifier
  - Executed directly?: Yes
  - Main functions: `train_one_epoch`, `evaluate`, `main`
  - Main classes: None
  - Output produced: Model `.pth` weights, JSON training logs

- Filename: `train_segmentor.py`
  - Extension: `.py`
  - Purpose: Training script for the segmentation models (U-Net, U-Net++)
  - Language: Python
  - Size: 6492 bytes
  - Dependencies: argparse, json, torch, tqdm, segmentation_models_pytorch
  - Imported by: None
  - Imports: os, argparse, json, torch, torch.nn, torch.utils.data.DataLoader, tqdm.tqdm, segmentation_models_pytorch, sys, src.config, src.utils, src.dataset, src.models.segmentor.BrainSegmentor
  - Executed directly?: Yes
  - Main functions: `train_one_epoch`, `evaluate`, `main`, `combined_loss`
  - Main classes: None
  - Output produced: Model `.pth` weights, JSON training logs

- Filename: `utils.py`
  - Extension: `.py`
  - Purpose: Utility functions for seeding, splitting patients, calculating metrics, and checkpoints
  - Language: Python
  - Size: 2081 bytes
  - Dependencies: numpy, torch, sklearn
  - Imported by: `src/evaluate.py`, `src/inference.py`, `src/train_classifier.py`, `src/train_segmentor.py`
  - Imports: os, random, numpy, torch, sklearn.model_selection.train_test_split, config.SEED
  - Executed directly?: No
  - Main functions: `set_seed`, `get_patient_split`, `dice_coef_metric`, `iou_metric`, `save_checkpoint`, `load_checkpoint`
  - Main classes: None
  - Output produced: Checkpoints

- Filename: `compare.py`
  - Extension: `.py`
  - Purpose: Master XAI script that generates Grad-CAM, Agreement, and Uncertainty maps
  - Language: Python
  - Size: 3043 bytes
  - Dependencies: argparse, numpy, torch, cv2, matplotlib, PIL
  - Imported by: `app.py`
  - Imports: os, sys, argparse, numpy, torch, cv2, matplotlib.pyplot, PIL.Image, src.config, src.dataset.get_val_transform, src.inference.load_ensembles, src.explainability.gradcam, src.explainability.segmentation_xai
  - Executed directly?: Yes
  - Main functions: `generate_full_explanation`
  - Main classes: None
  - Output produced: PNG heatmaps (`_gradcam.png`, `_agreement.png`, `_uncertainty.png`)

- Filename: `gradcam.py`
  - Extension: `.py`
  - Purpose: Implements Grad-CAM for visualizing classifier focus
  - Language: Python
  - Size: 2024 bytes
  - Dependencies: torch, numpy, cv2, matplotlib
  - Imported by: `src/explainability/compare.py`
  - Imports: torch, torch.nn.functional, numpy, cv2, matplotlib.pyplot
  - Executed directly?: No
  - Main functions: `show_cam_on_image`
  - Main classes: `GradCAM`
  - Output produced: Rendered heatmaps

- Filename: `segmentation_xai.py`
  - Extension: `.py`
  - Purpose: Implements segmentation Agreement Maps and Monte Carlo Dropout Uncertainty
  - Language: Python
  - Size: 2763 bytes
  - Dependencies: numpy, torch, cv2, matplotlib
  - Imported by: `src/explainability/compare.py`
  - Imports: numpy, torch, cv2, matplotlib.pyplot
  - Executed directly?: No
  - Main functions: `generate_agreement_map`, `visualize_agreement_map`, `mc_dropout_uncertainty`, `visualize_uncertainty`
  - Main classes: None
  - Output produced: Plotted variance and agreement maps

- Filename: `classifier.py`
  - Extension: `.py`
  - Purpose: PyTorch module defining the CNN classification architectures (timm wrapper)
  - Language: Python
  - Size: 1702 bytes
  - Dependencies: torch, timm
  - Imported by: `src/inference.py`, `src/train_classifier.py`
  - Imports: torch, torch.nn, timm
  - Executed directly?: No
  - Main functions: None
  - Main classes: `BrainClassifier`
  - Output produced: PyTorch Model definitions

- Filename: `segmentor.py`
  - Extension: `.py`
  - Purpose: PyTorch module defining the U-Net architectures (smp wrapper)
  - Language: Python
  - Size: 1249 bytes
  - Dependencies: torch, segmentation_models_pytorch
  - Imported by: `src/inference.py`, `src/train_segmentor.py`
  - Imports: segmentation_models_pytorch, torch.nn
  - Executed directly?: No
  - Main functions: None
  - Main classes: `BrainSegmentor`
  - Output produced: PyTorch Model definitions

- Filename: `explore_data.py`
  - Extension: `.py`
  - Purpose: Generates statistical plots and data distributions
  - Language: Python
  - Size: 18130 bytes
  - Dependencies: numpy, PIL, matplotlib
  - Imported by: None
  - Imports: os, glob, random, numpy, PIL.Image, collections.defaultdict, matplotlib, matplotlib.pyplot, matplotlib.gridspec, matplotlib.colors.LinearSegmentedColormap
  - Executed directly?: Yes
  - Main functions: `gather_stats`, `plot_class_distribution`, `plot_patient_distribution`, `plot_sample_positive`, `plot_sample_negative`, `plot_mask_area_distribution`, `plot_intensity_analysis`, `plot_summary_dashboard`
  - Main classes: None
  - Output produced: Analysis `.png` files in `results/exploration/`

- Filename: `generate_advanced_diagram.py`
  - Extension: `.py`
  - Purpose: Generates architecture diagram via Kroki API
  - Language: Python
  - Size: 3493 bytes
  - Dependencies: requests
  - Imported by: None
  - Imports: requests
  - Executed directly?: Yes
  - Main functions: None
  - Main classes: None
  - Output produced: `Full_Architecture.pdf`

- Filename: `generate_diagram.py`
  - Extension: `.py`
  - Purpose: Alternative diagram generator using matplotlib
  - Language: Python
  - Size: 1929 bytes
  - Dependencies: matplotlib
  - Imported by: None
  - Imports: matplotlib.pyplot, matplotlib.patches
  - Executed directly?: Yes
  - Main functions: `draw_box`, `draw_arrow`
  - Main classes: None
  - Output produced: `Architecture_Diagram.pdf`

- Filename: `generate_ppt.py`
  - Extension: `.py`
  - Purpose: Generates PowerPoint presentation automatically
  - Language: Python
  - Size: 8940 bytes
  - Dependencies: python-pptx
  - Imported by: None
  - Imports: pptx.Presentation, pptx.util, os
  - Executed directly?: Yes
  - Main functions: None
  - Main classes: None
  - Output produced: `Final_Presentation.pptx`

---------------------------------------------------------

# 5. Execution Flow
- Which file starts first: User runs `app.py` for inference or `train_classifier.py` / `train_segmentor.py` for training.
- Execution sequence (Web UI): 
  1. `app.py` initializes server, calls `load_ensembles()` from `src.inference`.
  2. Route `/` accepts POST with MRI image.
  3. Image saved to `static/uploads/`.
  4. Calls `run_inference()` from `src.inference`.
  5. `run_inference()` processes image through `ClassifierEnsemble`.
  6. If Tumor Probability > 0.5, processes image through `SegmentorEnsemble`.
  7. Calls `generate_full_explanation()` from `src.explainability.compare` to generate Grad-CAM, Agreement, and Uncertainty maps.
  8. Renders `index.html` with generated assets.
- Data flow: `.tif` upload -> Numpy Array -> Normalized Tensor -> PyTorch Models -> Raw Probabilities / Mask Tensors -> Numpy Masks -> Base64/PNG rendered in Browser.
- Training flow: 
  1. Split patients via `get_patient_split()`
  2. Initialize `BrainMRIDataset` and `DataLoader`
  3. Init Models (`BrainClassifier` or `BrainSegmentor`), Optimizer, Loss, and `torch.amp.GradScaler`
  4. Train loop (Epochs) calculating loss, performing backprop via AMP.
  5. Evaluate on Validation set.
  6. Save checkpoint if metrics improve.
- Saving/loading sequence: Models are saved via `save_checkpoint` to `checkpoints/` as `.pth` files. They are loaded back using `load_checkpoint` inside `load_ensembles`.

---------------------------------------------------------

# 6. Machine Learning / Deep Learning
- Models used: ResNet34, EfficientNet-B0, DenseNet121, U-Net, U-Net++
- Algorithms: Ensembling (Soft Voting, Mask Averaging), Monte Carlo Dropout
- Loss Functions: BCEWithLogitsLoss (Classification, pos_weight adjusted), DiceLoss + BCEWithLogitsLoss (Segmentation)
- Optimizers: AdamW
- Metrics: Accuracy, F1 Score, ROC AUC, Dice Score, Intersection over Union (IoU)
- Callbacks: Early Stopping (Manual Patience Counter), Best Model Checkpointing
- Schedulers: CosineAnnealingLR
- Activation Functions: ReLU (Grad-CAM), Sigmoid (Inference outputs)
- Transfer Learning Models: Pretrained ImageNet encoders (ResNet34, EfficientNet-B0, DenseNet121)
- Feature Engineering: None
- Preprocessing: Grayscale to RGB mapping (Classification backbone requirements), Normalization (dividing by 255.0)
- Postprocessing: Image extraction, Mask thresholding (>0.5), Bounding box calculation (cv2.boundingRect)
- Inference Pipeline: Sequential execution; Image -> Classifier Ensemble -> If Positive -> Segmentor Ensemble -> Heatmap Generation.
- Training Pipeline: AMP (Automatic Mixed Precision) driven, backbone freezing/unfreezing logic (frozen epoch 1, unfrozen epoch 6).

---------------------------------------------------------

# 7. Dataset Information
- Dataset name: Kaggle LGG MRI Segmentation Dataset
- Source: Kaggle (TCGA Dataset)
- Location: `kaggle_3m/`
- File format: `.tif` (Images and Masks), `.csv` (Clinical data)
- Image size: 256x256
- Number of classes: 2 (Tumor, No Tumor)
- Annotations: Binary Masks
- Splits: Train (80%), Validation (10%), Test (10%) at the Patient Level
- Augmentations: HorizontalFlip, VerticalFlip, Rotate, ElasticTransform, RandomBrightnessContrast
- Normalization: Mask divided by 255.0, Image divided by 255.0.

---------------------------------------------------------

# 8. Dependencies
- Extracted from: `requirements.txt`
  - `Flask` (Web framework - app.py)
  - `Werkzeug` (File handling - app.py)
  - `torch` (Deep Learning backend)
  - `torchvision` (CV Utils)
  - `timm` (Classification architectures)
  - `segmentation-models-pytorch` (Segmentation architectures)
  - `albumentations` (Augmentations)
  - `opencv-python-headless` (CV2 for bounding boxes and XAI image operations)
  - `Pillow` (Image loading)
  - `scikit-learn` (Metrics: F1, Acc, AUC)
  - `matplotlib` (Plotting and heatmaps)
  - `seaborn` (Confusion matrices)
  - `numpy` (Array operations)
  - `tqdm` (Progress bars)
  - `python-pptx` (PPT Generation)
  - `requests` (Diagram API generation)
  - Other generic packages (pandas, scipy, jupyter, etc.)

---------------------------------------------------------

# 9. Database
- Database used: None
- Schema: N/A
- Tables: N/A
- Relationships: N/A
- Queries: N/A
- ORM: N/A

---------------------------------------------------------

# 10. APIs
- Internal APIs: Flask Web App (`/` GET/POST route)
- External APIs: Kroki Diagram API (`https://kroki.io/mermaid/pdf` in `generate_advanced_diagram.py`)
- Authentication: None
- Environment Variables: `PYTHONHASHSEED`

---------------------------------------------------------

# 11. Configuration
- Configuration files: `src/config.py`
- Constants defined: 
  - `BASE_DIR`, `DATA_DIR`, `CHECKPOINT_DIR`, `RESULTS_DIR`
  - `DEVICE` (cuda or cpu)
  - `MIXED_PRECISION` (True)
  - `IMG_SIZE` (256)
  - `BATCH_SIZE` (16)
  - `NUM_WORKERS` (4)
  - `LR` (1e-4)
  - `EPOCHS` (50)
  - `PATIENCE` (10)
  - `SEED` (42)
  - `CLASSIFIERS` (List)
  - `SEGMENTORS` (List of dicts)

---------------------------------------------------------

# 12. Assets
- Images: User-uploaded `.png`, `.jpg`, `.tif` to `static/uploads/`, Generated Heatmaps to `static/xai/`
- Icons: None
- Videos: None
- Fonts: None
- Templates: `templates/index.html`
- Static files: `static/css/style.css`

---------------------------------------------------------

# 13. Deployment
- Docker: No
- Cloud Services: Unknown
- CI/CD: Unknown
- Instruction provided for local execution via `venv` in PowerShell.

---------------------------------------------------------

# 14. Architecture
- System Architecture: Monolithic Python Flask Application calling PyTorch scripts internally.
- Execution Flow Diagram: Client -> `app.py` -> `src/inference.py` -> `src/ensemble.py` -> `src/explainability/compare.py` -> HTML Render.

---------------------------------------------------------

# 15. Statistics
- Total folders: 119 (including 110 patients)
- Python files: 15
- JavaScript files: 0
- Configuration files: 1 (`src/config.py`)
- Classes: 4 (`BrainMRIDataset`, `ClassifierEnsemble`, `SegmentorEnsemble`, `BrainClassifier`, `BrainSegmentor`, `GradCAM`)
- Functions: ~25
- Models: 5 Base Architectures
- Datasets: 1 (Kaggle 3m)

---------------------------------------------------------

# 16. Missing Information
- Missing Model Weights: Weights are expected in `checkpoints/` but actual `.pth` files are not part of the read repository files.
- Missing API keys: None required.
- Unknown execution path: The specific external deployment environment (if any).
- Dataset Integrity: Some Kaggle metadata (like patient identifiers) exists, but exact dataset completeness (e.g. `data.csv` contents) was not thoroughly extracted.

---------------------------------------------------------

# 17. Final Project Specification
This factual specification document contains every identified script, framework, model, component, module dependency, and execution flow found within the "neurovision-mri" project directory as extracted from direct source code inspection. It is structured entirely of absolute facts derived from imports, syntax, and directory listings.
