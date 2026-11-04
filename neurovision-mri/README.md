# Brain Tumor MRI Segmentation & Explainable AI Pipeline

This repository contains a full PyTorch pipeline for classifying, localizing (segmenting), and visually explaining the presence of Lower-Grade Gliomas (LGG) in Brain MRI slices. 

The pipeline is intentionally built to support **Ensemble Learning** across both classification and segmentation architectures, and heavily utilizes **Mixed Precision** training to fit within stringent memory constraints (designed specifically for laptops with a 4GB VRAM NVIDIA RTX 3050 GPU).

---

## 🧠 Project Architecture

### 1. The Dataset (Kaggle LGG MRI)
* 110 patients containing 3,929 slices total.
* Data is strictly split on the **patient level** (no slices from a training patient leak into the validation or test sets).
* Strong spatial and intensity augmentations (via Albumentations) are applied identically across images and masks to combat the limited dataset size.

### 2. Classification Ensemble (Soft Voting)
Determines whether an MRI slice contains any tumor mass at all.
* **ResNet34:** A proven residual architecture for medical imaging.
* **EfficientNet-B0:** Compound scaling for high accuracy-per-parameter.
* **DenseNet121:** Dense feature reuse to pick up on micro-textures.

### 3. Segmentation Ensemble (Mask Averaging)
Draws the exact boundaries of the tumor (only executes if the classification ensemble predicts the presence of a tumor).
* **U-Net + ResNet34 Encoder**
* **U-Net + EfficientNet-B0 Encoder**
* **U-Net++ + DenseNet121 Encoder**

### 4. Explainable AI (XAI)
Provides clinical transparency.
* **Grad-CAM:** Generates heatmaps pointing to the regions that triggered the classifier.
* **Agreement Maps:** Highlights pixel-wise consensus across all segmentors.
* **Monte Carlo Dropout Uncertainty:** Maps the variance in multiple stochastic passes to visualize areas of model uncertainty.

### 5. Clinical Web Interface
A fully functional Flask web application that locally hosts the ensemble models. It provides a clean, medical-aesthetic interface for uploading MRI scans (supporting `.png`, `.jpg`, and `.tif`), instantly executing the full pipeline, and displaying the prediction confidence alongside all Explainable AI heatmaps.

---

## 🏆 Final Results

After training the complete ensemble, the pipeline achieved the following performance on the **unseen patient test set** (349 slices):

### Classification Performance
* **Accuracy:** 95.70%
* **F1 Score:** 94.62%
* **ROC AUC:** 98.15%

### Segmentation Performance
* **Average Dice Score (F1):** 88.33%
* **Average IoU Score:** 79.53%

> 💡 **View the full Showcase:** Open the `Showcase.html` or `Showcase.ipynb` file in this directory to see these metrics accompanied by visual Confusion Matrices and the **Explainable AI** heatmaps mapped directly onto sample MRI scans!

---

## 📚 The Ultimate Technical Book

This repository now includes a **50-page, production-quality LaTeX Technical Book** documenting every single aspect of this project for university defenses, peer reviews, or technical deep-dives. 

You can find the fully compiled PDF here: **[`project-book/main.pdf`](project-book/main.pdf)**

**The book covers:**
* Deep mathematical explanations of ResNet, EfficientNet, DenseNet, U-Net, and U-Net++ tensor flows.
* Architectural blueprints and directory structures.
* Exact loss functions (Dice + BCE) and clinical rationale.
* An interactive Q&A "Interview Cheat Sheet".
* A brutal, honest critique of the pipeline's weaknesses and future scope.

---

## 💻 Quick Start (Running Locally)

If you are cloning this repository to run on your own machine, follow these steps to get the Web Application up and running:

**1. Clone the repository**
```bash
git clone <your-github-repo-url>
cd "neurovision-mri"
```

**2. Create and activate a Virtual Environment (Python 3.12 recommended)**
```bash
# On Windows
python -m venv venv
.\venv\Scripts\activate

# On Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

**3. Install Dependencies**
```bash
pip install -r requirements.txt
```

**4. Run the Web Application**
```bash
python app.py
```
Once the console says `Running on http://127.0.0.1:8080`, open that link in your browser to start using the medical assessment tool!

---

## 🚀 How to Train & Run

Due to 4GB GPU limits, the models must be trained sequentially. 
Please read the **`GUIDE.md`** file located in this directory for exact, step-by-step terminal commands on how to run the training scripts, evaluate the ensemble, and test the inference pipeline.