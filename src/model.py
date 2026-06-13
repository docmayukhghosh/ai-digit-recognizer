import tensorflow as tf
from tensorflow.keras import layers, models
from typing import Tuple

def build_model(input_shape: Tuple[int, int, int] = (28, 28, 1), num_classes: int = 10) -> models.Sequential:
    """
    Builds and compiles a Convolutional Neural Network (CNN) for digit recognition.
    
    The network is designed using standard best practices:
    - Feature extraction: Conv2D layers with ReLU activations.
    - Spatial reduction: MaxPooling2D.
    - Regularization: Dropout to prevent overfitting.
    - Classification: Dense layers ending with a Softmax output layer.
    
    Args:
        input_shape (Tuple[int, int, int]): Dimensions of the input image (Height, Width, Channels).
                                            Default is (28, 28, 1) for MNIST.
        num_classes (int): Number of target classes. Default is 10 (digits 0-9).
        
    Returns:
        models.Sequential: A compiled Keras Sequential model.
    """
    model = models.Sequential(name="Digit_Recognizer_CNN")
    
    # 1. First Convolutional Layer
    # What it does: Slides 32 filters (each 3x3) over the image to detect simple patterns like edges/lines.
    # Why ReLU: Rectified Linear Unit, f(x) = max(0, x), adds non-linearity so the network can learn complex patterns.
    # Input shape is specified here to define the network's input entrypoint.
    model.add(layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape, name="conv2d_1"))
    
    # 2. Second Convolutional Layer
    # What it does: Slides 64 filters (each 3x3) to combine basic features from layer 1 into higher-level features (e.g., curves).
    model.add(layers.Conv2D(64, kernel_size=(3, 3), activation='relu', name="conv2d_2"))
    
    # 3. Max Pooling Layer
    # What it does: Reduces spatial dimensions (24x24 -> 12x12) by taking the maximum value in 2x2 windows.
    # Why it's needed: Reduces computational cost, reduces parameters (reducing overfitting), and makes the model robust to shifts.
    model.add(layers.MaxPooling2D(pool_size=(2, 2), name="max_pooling2d"))
    
    # 4. Dropout Layer (Regularization)
    # What it does: Randomly sets 25% of activations to 0 during training.
    # Why it's needed: Forces the network to learn redundant representations rather than relying on specific path connections.
    model.add(layers.Dropout(0.25, name="dropout_1"))
    
    # 5. Flatten Layer
    # What it does: Flattens the 3D feature map (12x12x64) into a 1D vector (9216 elements).
    # Why it's needed: Prepares data to enter the traditional feedforward neural network classifier.
    model.add(layers.Flatten(name="flatten"))
    
    # 6. Dense (Fully Connected) Layer
    # What it does: Connects all 9216 inputs to 128 nodes, learning final feature combinations.
    model.add(layers.Dense(128, activation='relu', name="dense_1"))
    
    # 7. Second Dropout Layer (Regularization)
    # What it does: Randomly drops 50% of the connections during training.
    # Why it's needed: Further prevents overfitting in the large fully connected layer.
    model.add(layers.Dropout(0.5, name="dropout_2"))
    
    # 8. Output Layer (Dense with Softmax)
    # What it does: 10 outputs corresponding to classes 0-9.
    # Why Softmax: Converts raw output values (logits) into a probability distribution (values between 0-1 that sum to 1.0).
    model.add(layers.Dense(num_classes, activation='softmax', name="output"))
    
    # Compile the model
    # Optimizer: Adam is an adaptive learning rate optimization algorithm (widely regarded as a default standard).
    # Loss: categorical_crossentropy measures the distance between the predicted probability and the true one-hot label.
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

if __name__ == "__main__":
    # Test building and summarizing the model
    model = build_model()
    model.summary()
