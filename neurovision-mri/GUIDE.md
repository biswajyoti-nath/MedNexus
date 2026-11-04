# Training and Inference Guide

Since deep learning models take significant time and hardware resources to train, especially on a laptop GPU like the RTX 3050 (4GB VRAM), we've designed this pipeline so you can train the models step-by-step.

Follow these instructions whenever you have free time to let your laptop crunch the numbers.

---

## 1. Preparation
Before running any script, you must open a terminal (PowerShell) inside your `neurovision-mri` folder and activate the Python environment we built:

```powershell
# Open PowerShell in your project folder and run:
.\environments\venv312\Scripts\activate
```
*(You will know it's activated when `(venv312)` appears at the start of your command prompt.)*

---

## 2. Train the Classifiers (Phase 1)
These models determine if an MRI slice has a tumor (`1`) or not (`0`). Train them one at a time. Each model will take roughly 30-50 minutes to complete its 50 epochs. 

Run these commands sequentially (wait for one to finish before starting the next):

```powershell
python src\train_classifier.py --model resnet34
python src\train_classifier.py --model efficientnet_b0
python src\train_classifier.py --model densenet121
```
*Where do they save?* `checkpoints/classification/`

---

## 3. Train the Segmentors (Phase 2)
These models trace the exact boundary of the tumor. They will only train on slices that actually contain a tumor.

Run these sequentially:

```powershell
python src\train_segmentor.py --arch Unet --encoder resnet34
python src\train_segmentor.py --arch Unet --encoder efficientnet-b0
python src\train_segmentor.py --arch UnetPlusPlus --encoder densenet121
```
*Where do they save?* `checkpoints/segmentation/`

---

## 4. Evaluate the Ensemble (Phase 3)
Once all 6 models have finished training, you can evaluate their combined power using the test set.

```powershell
python src\evaluate.py --split test
```
This will print out your Accuracy, F1 Score, Dice Score, and IoU. It will also generate a Confusion Matrix image in the `results/` folder.

---

## 5. Run Inference & XAI on a Single Image
To test the entire pipeline visually on a single MRI slice, run the inference script. It will run the image through all classifiers (to vote on if a tumor exists) and all segmentors (to draw the bounding box).

```powershell
python src\inference.py --image "kaggle_3m\TCGA_CS_4941_19960909\TCGA_CS_4941_19960909_11.tif" --save results/demo.png
```

To see the **Explainable AI (XAI)** (Grad-CAM heatmaps, Uncertainty Maps, and Model Agreement Maps), run:
```powershell
python src\explainability\compare.py --image "kaggle_3m\TCGA_CS_4941_19960909\TCGA_CS_4941_19960909_11.tif" --save_dir results/xai
```

---

## 6. Launching the Clinical Web Application
Instead of running inference from the terminal, you can use the beautifully designed, single-page web interface to test images in your browser:

```powershell
python app.py
```
* Wait for the console to say `Running on http://127.0.0.1:8080`.
* Open **http://127.0.0.1:8080** in your web browser (Chrome, Edge, etc.).
* You can upload any `.tif`, `.png`, or `.jpg` MRI image (there are samples provided in the `sample_images/` folder).
* The web app will automatically execute the full pipeline and display the results, including the XAI heatmaps, directly on the page!
