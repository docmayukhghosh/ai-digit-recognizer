import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import utils
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from typing import Tuple, Dict, Any






# Import model builder and preprocessing utilities
from src.model import build_model
from src.preprocess import augment_image

def load_and_preprocess_mnist(val_split: float = 0.1, augment: bool = False) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Downloads MNIST dataset from Keras, normalizes pixels, performs validation split, 
    and applies data augmentation if requested.
    
    Args:
        val_split (float): Fraction of training data to use for validation.
        augment (bool): Whether to apply data augmentation to the training images.
        
    Returns:
        Tuple of training, validation, and test datasets:
        (x_train, y_train), (x_val, y_val), (x_test, y_test)
    """
    print("Loading MNIST dataset...")
    # Load raw data from tensorflow keras
    (x_train_raw, y_train_raw), (x_test_raw, y_test_raw) = tf.keras.datasets.mnist.load_data()
    
    # 1. Normalize pixel values from [0, 255] to [0.0, 1.0]
    x_train_raw = x_train_raw.astype("float32") / 255.0
    x_test_raw = x_test_raw.astype("float32") / 255.0
    
    # 2. Reshape to include channel dimension (Height, Width, Channels) -> (28, 28, 1)
    x_train_raw = np.expand_dims(x_train_raw, axis=-1)
    x_test_raw = np.expand_dims(x_test_raw, axis=-1)
    
    # 3. Validation Split
    # Since we want to display class distributions, we will manually split and check
    num_train_samples = len(x_train_raw)
    num_val_samples = int(num_train_samples * val_split)
    
    # Shuffle training indices before splitting
    indices = np.arange(num_train_samples)
    np.random.seed(42)  # For reproducibility
    np.random.shuffle(indices)
    
    val_indices = indices[:num_val_samples]
    train_indices = indices[num_val_samples:]
    
    x_train = x_train_raw[train_indices]
    y_train_lbls = y_train_raw[train_indices]
    
    x_val = x_train_raw[val_indices]
    y_val_lbls = y_train_raw[val_indices]
    
    x_test = x_test_raw
    y_test_lbls = y_test_raw
    
    # 4. Optional Data Augmentation
    # If set, we apply our manual image augmentations to train images to double the training set
    if augment:
        print("Applying data augmentation to training images...")
        augmented_x = []
        augmented_y = []
        for i in range(len(x_train)):
            augmented_x.append(augment_image(x_train[i]))
            augmented_y.append(y_train_lbls[i])
            
        x_train = np.concatenate([x_train, np.array(augmented_x)], axis=0)
        y_train_lbls = np.concatenate([y_train_lbls, np.array(augmented_y)], axis=0)
        print(f"Data augmented! New training size: {len(x_train)}")

    # 5. One-hot encode the target labels
    # E.g., digit 3 -> [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    y_train = utils.to_categorical(y_train_lbls, 10)
    y_val = utils.to_categorical(y_val_lbls, 10)
    y_test = utils.to_categorical(y_test_lbls, 10)
    
    print(f"Training set: {x_train.shape[0]} images")
    print(f"Validation set: {x_val.shape[0]} images")
    print(f"Test set: {x_test.shape[0]} images")
    
    # Save statistics of class distribution
    plot_class_distribution(y_train_lbls, "Training")
    
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


def plot_class_distribution(labels: np.ndarray, subset_name: str) -> None:
    """
    Plots and saves the class distribution for a subset of data to ensure balanced splits.
    """
    unique, counts = np.unique(labels, return_counts=True)
    df = pd.DataFrame({'Digit': unique, 'Count': counts})
    
    plt.figure(figsize=(8, 4))
    plt.bar(df['Digit'], df['Count'], color='#4F46E5', edgecolor='#3730A3')
    plt.xlabel('Digit Class')
    plt.ylabel('Number of Samples')
    plt.title(f'MNIST Digit Class Distribution in {subset_name} Set')
    plt.xticks(range(10))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Create data folder if not exists
    os.makedirs("data", exist_ok=True)
    plt.savefig(f"data/{subset_name.lower()}_distribution.png", dpi=100)
    plt.close()
    
    print(f"{subset_name} class counts:")
    for digit, count in zip(unique, counts):
        print(f"  Digit {digit}: {count} samples ({count/len(labels)*100:.2f}%)")


def plot_training_history(history: tf.keras.callbacks.History, save_path: str = "data/training_history.png") -> None:
    """
    Generates and saves curves showing training vs validation loss and accuracy.
    """
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(1, len(acc) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy Plot
    ax1.plot(epochs_range, acc, label='Training Accuracy', color='#4F46E5', linewidth=2)
    ax1.plot(epochs_range, val_acc, label='Validation Accuracy', color='#10B981', linewidth=2)
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Training vs Validation Accuracy')
    ax1.legend(loc='lower right')
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    # Loss Plot
    ax2.plot(epochs_range, loss, label='Training Loss', color='#EF4444', linewidth=2)
    ax2.plot(epochs_range, val_loss, label='Validation Loss', color='#F59E0B', linewidth=2)
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss')
    ax2.set_title('Training vs Validation Loss')
    ax2.legend(loc='upper right')
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    plt.suptitle("Model Training Diagnostics", fontsize=16, fontweight='bold', color='#1F2937')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Training history graphs saved to {save_path}")


def train_model(epochs: int = 15, batch_size: int = 128, val_split: float = 0.1, augment: bool = False, model_name: str = "digit_recognizer_model.keras") -> tf.keras.callbacks.History:
    """
    Loads data, compiles model, initializes training callbacks, runs fit process, 
    and saves the trained weights.
    
    Args:
        epochs (int): Max number of epochs to train.
        batch_size (int): Size of training batches.
        val_split (float): Portion of training data to use for validation.
        augment (bool): Whether to apply data augmentation.
        model_name (str): Path filename to save the model.
        
    Returns:
        tf.keras.callbacks.History: Training history dictionary object.
    """
    # Create directories if they don't exist
    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    
    model_path = os.path.join("models", model_name)
    
    # 1. Load Preprocessed Data
    (x_train, y_train), (x_val, y_val), (_, _) = load_and_preprocess_mnist(val_split=val_split, augment=augment)
    
    # 2. Build the CNN Model
    model = build_model(input_shape=(28, 28, 1), num_classes=10)
    
    # 3. Setup Callbacks
    # Early Stopping halts training if validation loss doesn't improve for 5 epochs.
    early_stop = EarlyStopping(
        monitor='val_loss', 
        patience=5, 
        restore_best_weights=True,
        verbose=1
    )
    
    # ModelCheckpoint saves only the best model weights based on validation loss.
    checkpoint = ModelCheckpoint(
        filepath=model_path,
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
    
    # ReduceLROnPlateau drops learning rate by a factor of 10 if val loss plateaus for 3 epochs.
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )
    
    callbacks_list = [early_stop, checkpoint, reduce_lr]
    
    # 4. Train the Model
    print(f"Starting model training for up to {epochs} epochs...")
    history = model.fit(
        x_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_val, y_val),
        callbacks=callbacks_list,
        verbose=1
    )
    
    # 5. Generate and Save Training History Visualizations
    plot_training_history(history)
    
    print(f"Training completed successfully! Best model saved to: {model_path}")
    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Digit Recognition CNN model from scratch.")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--val_split", type=float, default=0.1, help="Validation split fraction")
    parser.add_argument("--augment", action="store_true", help="Apply data augmentation during training")
    parser.add_argument("--model_name", type=str, default="digit_recognizer_model.keras", help="Filename for saved model")
    
    args = parser.parse_args()
    
    train_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        val_split=args.val_split,
        augment=args.augment,
        model_name=args.model_name
    )
