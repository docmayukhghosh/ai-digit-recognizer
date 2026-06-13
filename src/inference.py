import os
import argparse
import numpy as np
import cv2
import tensorflow as tf
from typing import Tuple, Dict, Any, Optional

# Import preprocessing
from src.preprocess import preprocess_image_for_inference

class DigitRecognizer:
    """
    Inference pipeline for digit recognition.
    Loads the trained Keras CNN model and provides a method to predict digits 
    from raw numpy image arrays.
    """
    def __init__(self, model_path: str = "models/digit_recognizer_model.keras"):
        """
        Initializes the classifier by loading the saved Keras model.
        
        Args:
            model_path (str): Path to the saved Keras model (.keras file).
        """
        self.model_path = model_path
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file not found at: {model_path}. "
                f"Please train the model first by running train_pipeline.py."
            )
        print(f"Loading trained model from {model_path}...")
        self.model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully!")

    def predict(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Preprocesses a raw input image, runs it through the CNN model, and 
        returns predictions and explainability data.
        
        Args:
            image (np.ndarray): Input image array (any size, color/grayscale).
            
        Returns:
            Dict[str, Any]: A dictionary containing:
                - "predicted_digit" (int): The recognized digit (0-9).
                - "confidence" (float): The confidence percentage (0-100).
                - "probabilities" (np.ndarray): Probability list across classes 0-9.
                - "preprocessed_image" (np.ndarray): The 28x28 grayscale canvas.
                - "debug_image" (np.ndarray): OpenCV visual panel for pipeline analysis.
        """
        # 1. Preprocess the raw image using OpenCV (grayscale, threshold, invert, center)
        preprocessed, debug_img = preprocess_image_for_inference(image, debug=True)
        
        # 2. Add batch dimension: (28, 28, 1) -> (1, 28, 28, 1)
        # Keras model prediction expects a batch of inputs
        input_batch = np.expand_dims(preprocessed, axis=0)
        
        # 3. Model Inference
        # Returns a 2D array of shape (1, 10) representing probabilities for digits 0-9
        probs_batch = self.model.predict(input_batch, verbose=0)
        probabilities = probs_batch[0]  # Extract first sample in batch (shape: 10,)
        
        # 4. Get Predicted Class and Confidence
        predicted_digit = int(np.argmax(probabilities))
        confidence = float(probabilities[predicted_digit]) * 100.0
        
        return {
            "predicted_digit": predicted_digit,
            "confidence": confidence,
            "probabilities": probabilities,
            "preprocessed_image": preprocessed,
            "debug_image": debug_img
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on a custom digit image file.")
    parser.add_argument("image_path", type=str, help="Path to the input image file")
    parser.add_argument("--model", type=str, default="models/digit_recognizer_model.keras", help="Path to the Keras model file")
    args = parser.parse_args()
    
    # Check if image path exists
    if not os.path.exists(args.image_path):
        print(f"Error: Input image file '{args.image_path}' not found.")
        exit(1)
        
    # Read the image
    img = cv2.imread(args.image_path)
    if img is None:
        print(f"Error: Failed to load image '{args.image_path}'. Make sure it is a valid image.")
        exit(1)
        
    # Convert BGR (OpenCV default) to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    try:
        recognizer = DigitRecognizer(model_path=args.model)
        result = recognizer.predict(img_rgb)
        
        print("\n=== Prediction Results ===")
        print(f"Predicted Digit: {result['predicted_digit']}")
        print(f"Confidence:      {result['confidence']:.2f}%")
        print("\nClass Probabilities:")
        for digit, prob in enumerate(result['probabilities']):
            print(f"  Digit {digit}: {prob*100:6.2f}% " + "#" * int(prob * 30))
            
    except Exception as e:
        print(f"An error occurred: {e}")
