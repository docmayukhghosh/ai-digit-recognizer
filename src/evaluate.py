import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import itertools
from typing import Tuple, List

def load_test_data() -> Tuple[np.ndarray, np.ndarray]:
    """Loads and preprocesses test images and labels from MNIST."""
    _, (x_test_raw, y_test_raw) = tf.keras.datasets.mnist.load_data()
    x_test = x_test_raw.astype("float32") / 255.0
    x_test = np.expand_dims(x_test, axis=-1)
    return x_test, y_test_raw

def plot_confusion_matrix(cm: np.ndarray, classes: List[str], save_path: str = "data/confusion_matrix.png") -> None:
    """
    Plots a highly detailed and stylized confusion matrix using Matplotlib.
    """
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Digit Classifier Confusion Matrix', fontsize=14, fontweight='bold', pad=15)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=10)
    plt.yticks(tick_marks, classes, fontsize=10)

    # Threshold for text color contrast
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black",
                 fontsize=11)

    plt.tight_layout()
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.grid(False)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to {save_path}")

def plot_misclassified_examples(x_test: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, y_probs: np.ndarray, save_path: str = "data/misclassified_samples.png") -> None:
    """
    Identifies misclassifications and plots a 3x3 grid of samples to analyze errors.
    """
    # Find indices where prediction is wrong
    wrong_indices = np.where(y_true != y_pred)[0]
    
    if len(wrong_indices) == 0:
        print("Wow! Zero misclassifications found on test data.")
        return
        
    print(f"Total misclassifications on test set: {len(wrong_indices)} / {len(y_true)} ({len(wrong_indices)/len(y_true)*100:.2f}%)")
    
    # Pick up to 9 random indices from misclassified images
    num_to_plot = min(9, len(wrong_indices))
    selected_indices = np.random.choice(wrong_indices, num_to_plot, replace=False)
    
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    axes = axes.ravel()
    
    for i, idx in enumerate(selected_indices):
        img = x_test[idx].squeeze()
        true_lbl = y_true[idx]
        pred_lbl = y_pred[idx]
        conf = y_probs[idx][pred_lbl] * 100
        
        axes[i].imshow(img, cmap='gray')
        axes[i].axis('off')
        axes[i].set_title(f"True: {true_lbl} | Pred: {pred_lbl}\nConf: {conf:.1f}%", 
                          color='red', fontsize=10, fontweight='bold')
                          
    plt.suptitle("Analysis of Misclassified Samples", fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Misclassified samples plot saved to {save_path}")

def evaluate_model(model_name: str = "digit_recognizer_model.keras") -> None:
    """
    Loads saved model, evaluates test metrics, outputs classification reports, 
    and saves diagnostic plots.
    """
    model_path = os.path.join("models", model_name)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}. Please run training script first.")
        
    print(f"Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path)
    
    # Load test data
    x_test, y_test_true = load_test_data()
    
    # Run predictions
    print("Evaluating on test set...")
    y_test_onehot = tf.keras.utils.to_categorical(y_test_true, 10)
    loss, accuracy = model.evaluate(x_test, y_test_onehot, verbose=0)
    print(f"\n--- Test Results ---")
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy*100:.2f}%")
    
    y_probs = model.predict(x_test, verbose=0)
    y_pred = np.argmax(y_probs, axis=1)
    
    # 1. Generate classification report (precision, recall, f1-score per class)
    report = classification_report(y_test_true, y_pred, target_names=[str(i) for i in range(10)], output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    
    print("\nClassification Report:")
    print(report_df.round(4).to_string())
    
    # Save report as csv for future reference
    os.makedirs("data", exist_ok=True)
    report_df.to_csv("data/classification_report.csv")
    
    # 2. Confusion Matrix
    cm = confusion_matrix(y_test_true, y_pred)
    plot_confusion_matrix(cm, [str(i) for i in range(10)])
    
    # 3. Misclassifications Analysis
    plot_misclassified_examples(x_test, y_test_true, y_pred, y_probs)
    
    # Save a short summary report
    summary_path = "data/evaluation_summary.md"
    with open(summary_path, "w") as f:
        f.write("# Model Evaluation Summary\n\n")
        f.write(f"- **Model Path**: `{model_path}`\n")
        f.write(f"- **Test Accuracy**: {accuracy*100:.2f}%\n")
        f.write(f"- **Test Loss**: {loss:.4f}\n\n")
        f.write("## Per-Class Metrics\n\n")
        f.write(report_df.round(4).to_markdown())
        f.write("\n\n## Insights & Errors\n")
        f.write("The confusion matrix and misclassified examples plots have been generated and placed in the `/data/` folder.\n")
        f.write("Common errors in MNIST usually involve digits with overlapping structural styles, such as:\n")
        f.write("- **4 vs 9**: Incomplete loops at the top of 9s can resemble 4s.\n")
        f.write("- **3 vs 8**: Missing connections in 8s can resemble 3s.\n")
        f.write("- **7 vs 1**: Slanted lines can cause confusion between 7s and 1s.\n")
        
    print(f"Written text evaluation report to {summary_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the trained CNN model on MNIST test set.")
    parser.add_argument("--model_name", type=str, default="digit_recognizer_model.keras", help="Model filename in models/ folder")
    args = parser.parse_args()
    
    evaluate_model(model_name=args.model_name)
