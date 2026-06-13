import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import models
from typing import List, Tuple, Dict

def get_intermediate_activations(model: tf.keras.Model, image: np.ndarray, layer_names: List[str]) -> List[np.ndarray]:
    """
    Creates an activation model that outputs the feature maps of specified intermediate layers
    for a given input image.
    
    Args:
        model (tf.keras.Model): The trained digit recognition model.
        image (np.ndarray): Preprocessed image of shape (1, 28, 28, 1).
        layer_names (List[str]): Names of layers to extract feature maps from.
        
    Returns:
        List[np.ndarray]: A list of activations (feature maps) from the requested layers.
    """
    # Extract the output tensors of the requested layers
    layer_outputs = [model.get_layer(name).output for name in layer_names]
    
    # Create a model that returns these activation tensors, given the model input
    activation_model = models.Model(inputs=model.inputs, outputs=layer_outputs)
    
    # Predict on the single image to obtain the activations
    # Predict returns a list of arrays if layer_outputs contains multiple layers
    activations = activation_model.predict(image, verbose=0)
    
    # If only one layer was requested, wrap it in a list for consistent interface
    if len(layer_names) == 1:
        activations = [activations]
        
    return activations

def plot_feature_maps_to_fig(activation: np.ndarray, layer_name: str, max_filters: int = 16) -> plt.Figure:
    """
    Plots the feature maps of a single layer into a Matplotlib Figure.
    
    Args:
        activation (np.ndarray): Activation tensor of shape (1, height, width, num_filters).
        layer_name (str): The name of the layer (for the plot title).
        max_filters (int): Maximum number of filter feature maps to display (default 16).
        
    Returns:
        plt.Figure: Matplotlib figure containing the filter grid.
    """
    # Remove batch dimension: (height, width, num_filters)
    feature_maps = activation[0]
    num_filters = feature_maps.shape[-1]
    
    # Cap the number of filters we plot to avoid overly cluttered figures
    n_features = min(num_filters, max_filters)
    
    # Calculate grid size (e.g. 4x4 grid for 16 filters)
    grid_size = int(np.ceil(np.sqrt(n_features)))
    
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(8, 8))
    fig.suptitle(f"Feature Maps from Layer: '{layer_name}'\n(Showing first {n_features} filters)", 
                 fontsize=12, fontweight='bold', color='#1F2937')
    
    # Flatten the axes array for easy iteration
    axes_flat = axes.ravel() if isinstance(axes, np.ndarray) else [axes]
    
    for i in range(len(axes_flat)):
        if i < n_features:
            # Get the feature map of filter i
            f_map = feature_maps[:, :, i]
            
            # Post-process the feature map to make it visually clear
            # Normalize to 0-255 range for better visualization contrast
            if f_map.max() > 0:
                f_map = (f_map - f_map.min()) / (f_map.max() - f_map.min())
                
            axes_flat[i].imshow(f_map, cmap='viridis')
            axes_flat[i].set_title(f"Filter {i+1}", fontsize=8)
        
        # Hide axes ticks for cleanliness
        axes_flat[i].axis('off')
        
    plt.tight_layout()
    return fig

def plot_probability_distribution_to_fig(probabilities: np.ndarray) -> plt.Figure:
    """
    Generates a horizontal bar chart displaying prediction probabilities for digits 0-9.
    
    Args:
        probabilities (np.ndarray): Array of 10 float probabilities summing to 1.0.
        
    Returns:
        plt.Figure: Matplotlib figure containing the probability bar chart.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    digits = [str(i) for i in range(10)]
    
    # Find the index of the highest probability
    max_idx = np.argmax(probabilities)
    
    # Create custom color array (highlight the predicted class in purple/indigo)
    colors = ['#E5E7EB'] * 10  # light grey for rest
    colors[max_idx] = '#4F46E5'  # indigo for predicted
    
    bars = ax.barh(digits, probabilities, color=colors, edgecolor='#9CA3AF', height=0.6)
    
    # Add values on top of bars
    for bar in bars:
        width = bar.get_width()
        if width > 0.01:
            ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{width*100:.1f}%', 
                    va='center', ha='left', fontsize=9, fontweight='bold',
                    color='#374151')
            
    ax.set_xlim(0, 1.15)  # Leave room for text labels
    ax.set_xlabel('Probability / Confidence Score', fontsize=10)
    ax.set_ylabel('Digit Class', fontsize=10)
    ax.set_title('Prediction Probability Distribution', fontsize=12, fontweight='bold', pad=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='x', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    return fig
