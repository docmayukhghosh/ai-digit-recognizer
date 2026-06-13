---
title: Ai Digit Recognizer
emoji: 📊
colorFrom: red
colorTo: purple
sdk: gradio
sdk_version: 4.44.1
python_version: '3.9'
app_file: app.py
pinned: false
short_description: A interactive Deep Learning visualization for numbers
---

# 🧠 Deep Learning Digit Recognizer (MNIST & Real-world Photo Classifier)

A complete, end-to-end Machine Learning project written in Python using **TensorFlow/Keras**, **OpenCV**, and **Gradio** to recognize handwritten digits (0-9) from canvas drawings or custom uploaded photos. 

This project is beginner-friendly, fully documented, and designed as a learning resource to demonstrate:
1. Building a Convolutional Neural Network (CNN) from scratch.
2. Building an OpenCV image preprocessing pipeline for real-world photos.
3. Evaluating classification performance (F1-score, Precision, Recall, Confusion Matrix).
4. Explaining model decisions using intermediate feature maps.

---

## 📂 Project Directory Structure

```
digit_recognition_project/
├── data/                    # Class distribution graphs and validation curves
├── models/                  # Saved Keras model files (.keras format)
├── src/                     # Core source code
│   ├── __init__.py          # Marks src as a Python package
│   ├── preprocess.py        # OpenCV image preprocessing & binarization
│   ├── model.py             # CNN network architecture and documentation
│   ├── train.py             # Training loop, data splits, and callbacks
│   ├── evaluate.py          # Testing metrics and confusion matrix generator
│   ├── explain.py           # Feature map activation extraction
│   └── app.py               # Gradio web interface layout
├── requirements.txt         # Package dependencies
├── README.md                # This instructions document
├── train_pipeline.py        # Wrapper entrypoint to train the model
├── evaluate_pipeline.py     # Wrapper entrypoint to evaluate the model
└── run_app.py               # Wrapper entrypoint to start the Gradio app
```

---

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.9+ installed on your system.

### 1. Clone or Open Project Directory
Navigate to the project directory in your terminal:
```bash
cd C:\Users\USER\.gemini\antigravity\scratch\digit_recognition_project
```

### 2. Install Dependencies
Install the required packages using the Python Launcher:
```bash
py -m pip install --user -r requirements.txt
```

---

## 🚀 Running the Project

### Step 1: Train the Model
Train the CNN model from scratch on the MNIST dataset. By default, it runs for 15 epochs with a batch size of 128 and split out 10% validation data.
```bash
py train_pipeline.py
```
*Optional Augmentation*: To train the model with data augmentations (random rotations, shifts, zooms) to increase robustness on real-world photo distortions, run:
```bash
py train_pipeline.py --augment
```
*Outputs*:
- Saves the best-performing model to `models/digit_recognizer_model.keras`.
- Generates data distribution and training history plots in the `data/` directory.

### Step 2: Evaluate the Model
Run evaluations on the test set (10,000 unseen images) to compute classification performance metrics:
```bash
py evaluate_pipeline.py
```
*Outputs*:
- Prints a detailed classification report (Precision, Recall, F1-Score per digit).
- Saves a **Confusion Matrix** plot to `data/confusion_matrix.png`.
- Saves a **Misclassification Analysis Grid** to `data/misclassified_samples.png`.
- Saves a summary markdown report to `data/evaluation_summary.md`.

### Step 3: Launch the Gradio Web App
Run the interactive user interface:
```bash
py run_app.py
```
*Outputs*:
- Starts a local web server at `http://127.0.0.1:7860`.
- Open the link in a web browser to draw digits directly on a sketchpad, upload camera images, and view live classification logs, confidence distributions, and CNN feature maps.

---

## 🧠 Model Architecture & Hyperparameters

The model is a Convolutional Neural Network (CNN) built using Keras:
- **Conv2D (32 filters, 3x3 kernel, ReLU)**: Learns simple borders and edge filters.
- **Conv2D (64 filters, 3x3 kernel, ReLU)**: Combines borders to form curves and loops.
- **MaxPooling2D (2x2)**: Downsamples the feature maps, reducing parameters and noise.
- **Dropout (25%)**: Disables connections to prevent overfitting.
- **Flatten**: Converts 2D representations into a 1D vector.
- **Dense (128 units, ReLU)**: Fully connected layer that learns combinations of complex features.
- **Dropout (50%)**: Disables connections for final stage regularization.
- **Dense (10 units, Softmax)**: Outputs the class probability distribution for digits 0-9.

### Hyperparameters:
- **Optimizer**: Adam (adaptive learning rate)
- **Loss**: Categorical Crossentropy
- **Callbacks**:
  - `EarlyStopping`: stops training if validation loss plateaus for 5 epochs.
  - `ModelCheckpoint`: only saves the model if validation loss improves.
  - `ReduceLROnPlateau`: decreases learning rate if loss plateaus for 3 epochs.

---

## 📷 Tips for High Accuracy on Phone-Camera Photos

The MNIST dataset has clean white lines on a pure black background. Real-world photos have shadows, lighting changes, camera skew, and noise. We've implemented an OpenCV preprocessing pipeline to counter this, but for the best results:

1. **Sharp Contrast**: Draw your digit with a thick dark marker (like a Sharpie) on a plain white paper.
2. **Even Lighting**: Avoid shadows falling across the page. Shadows can confuse adaptive threshold algorithms into detecting the shadow edge as part of the digit.
3. **Square & Close Crop**: Hold the camera directly above the digit (avoid heavy angles) and crop closely to the digit before uploading.
4. **Drawing Canvas**: In the drawing tool, draw your digit in the center of the box using continuous, thick lines.

### Further Optimization Ideas:
If you want to modify this project to reach higher accuracy on photographed images, try:
- **Custom fine-tuning**: Collect 100-200 pictures of digits photographed with your own phone, label them, and fine-tune your model on this dataset.
- **Thicker Stroke Augmentation**: Add dilation/erosion filters during training augmentation to simulate various pen widths.
