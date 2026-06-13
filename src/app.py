import os
import numpy as np
import matplotlib.pyplot as plt
import gradio as gr
import tensorflow as tf

# Import internal modules
from src.inference import DigitRecognizer
from src.explain import (
    get_intermediate_activations, 
    plot_feature_maps_to_fig, 
    plot_probability_distribution_to_fig
)

# Initialize the digit recognizer global instance
MODEL_PATH = "models/digit_recognizer_model.keras"
recognizer = None

def get_recognizer():
    global recognizer
    if recognizer is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model file not found at: {MODEL_PATH}. "
                "Please run model training before launching the web app."
            )
        recognizer = DigitRecognizer(model_path=MODEL_PATH)
    return recognizer

def process_and_predict(image) -> tuple:
    """
    Main prediction logic for the Gradio interface.
    Takes a drawing canvas input or upload image, runs inference, 
    and returns predictions, charts, and debug visualizations.
    """
    # Guard case: Check if image is empty or black (e.g., cleared canvas)
    if image is None:
        return (
            "Draw or upload a digit first!", 
            None, 
            None, 
            None, 
            None
        )
        
    # If the input is from Gradio sketchpad, it could be a dictionary in newer versions:
    # { "background": ..., "layers": ..., "composite": ... }
    # We want to extract the drawn image array
    if isinstance(image, dict):
        if "composite" in image and image["composite"] is not None:
            image = image["composite"]
        elif "layers" in image and len(image["layers"]) > 0:
            image = image["layers"][0]
        else:
            return "Empty drawing!", None, None, None, None

    # Verify we have a valid numpy array
    if not isinstance(image, np.ndarray):
        return "Invalid image format!", None, None, None, None
        
    # Check if the canvas is entirely empty (all black or all white)
    if image.size == 0 or np.all(image == 0) or np.all(image == 255):
         return "Canvas is empty. Draw a digit!", None, None, None, None

    try:
        rec = get_recognizer()
        
        # 1. Run inference
        result = rec.predict(image)
        
        pred_digit = result["predicted_digit"]
        confidence = result["confidence"]
        probabilities = result["probabilities"]
        preprocessed = result["preprocessed_image"]
        debug_image = result["debug_image"]
        
        # 2. Format output text
        output_txt = (
            f"<div style='text-align: center; padding: 10px; border-radius: 8px; "
            f"background-color: #EEF2F6; border: 1px solid #CBD5E1;'>"
            f"<h2 style='margin: 0; color: #1E293B;'>Predicted Digit</h2>"
            f"<h1 style='margin: 5px 0; font-size: 72px; color: #4F46E5;'>{pred_digit}</h1>"
            f"<p style='margin: 0; font-size: 18px; font-weight: bold; color: #10B981;'>"
            f"Confidence: {confidence:.2f}%</p>"
            f"</div>"
        )
        
        # 3. Generate probability bar chart
        prob_fig = plot_probability_distribution_to_fig(probabilities)
        
        # 4. Generate Explainability Feature Maps (from Conv layers)
        # Reshape preprocessed image for activation model: (28, 28, 1) -> (1, 28, 28, 1)
        image_batch = np.expand_dims(preprocessed, axis=0)
        
        # Extract activations
        layer_names = ["conv2d_1", "conv2d_2"]
        activations = get_intermediate_activations(rec.model, image_batch, layer_names)
        
        # Plot feature maps
        fig_conv1 = plot_feature_maps_to_fig(activations[0], "conv2d_1 (Layer 1)", max_filters=16)
        fig_conv2 = plot_feature_maps_to_fig(activations[1], "conv2d_2 (Layer 2)", max_filters=16)
        
        return (
            output_txt, 
            prob_fig, 
            debug_image, 
            fig_conv1, 
            fig_conv2
        )
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Error during classification: {str(e)}", None, None, None, None

# Premium Custom CSS Styling
custom_css = """
body {
    background-color: #F8FAFC !important;
    font-family: 'Inter', sans-serif !important;
}
.gradio-container {
    max-width: 1200px !important;
    margin: 40px auto !important;
    padding: 20px !important;
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1), 0 8px 10px -6px rgba(0, 0, 0, 0.1) !important;
    border-radius: 16px !important;
    background-color: #FFFFFF !important;
}
.header-box {
    text-align: center;
    margin-bottom: 30px;
    padding-bottom: 20px;
    border-bottom: 2px solid #E2E8F0;
}
.header-box h1 {
    color: #1E293B;
    font-size: 32px;
    font-weight: 800;
    margin-bottom: 5px;
}
.header-box p {
    color: #64748B;
    font-size: 16px;
}
"""

def build_gradio_app() -> gr.Blocks:
    """
    Assembles the Gradio Blocks UI layout.
    """
    with gr.Blocks(css=custom_css, title="AI Digit Recognizer") as demo:
        # Header Section
        gr.HTML(
            "<div class='header-box'>"
            "<h1>🧠 Deep Learning Digit Recognizer</h1>"
            "<p>Draw a digit or upload a photo. The Convolutional Neural Network (CNN) "
            "will classify it and show what features it extracts at intermediate layers.</p>"
            "</div>"
        )
        
        with gr.Row():
            # Left Column - Inputs
            with gr.Column(scale=1):
                gr.Markdown("### 📥 Choose Input Method")
                
                with gr.Tab("Draw Canvas"):
                    draw_input = gr.Sketchpad(
                        label="Draw a single digit (0-9)",
                        type="numpy"
                    )
                    draw_btn = gr.Button("Classify Sketch", variant="primary")
                    
                with gr.Tab("Upload Photo"):
                    file_input = gr.Image(
                        label="Upload a photo of a digit",
                        type="numpy"
                    )
                    file_btn = gr.Button("Classify Upload", variant="primary")
                
                gr.Markdown(
                    "#### 💡 Tips for High Accuracy on Phone Photos:\n"
                    "1. Draw the digit clearly in the center of the frame.\n"
                    "2. Avoid shadows and complex backgrounds.\n"
                    "3. Ensure the digit contrast is sharp (dark pen on white paper)."
                )

            # Center Column - Output & Probability
            with gr.Column(scale=1):
                gr.Markdown("### 📊 Classification Output")
                output_html = gr.HTML(value="<div style='text-align: center; color: #64748B;'>Upload or draw an image, then click Classify.</div>")
                output_chart = gr.Plot(label="Confidence Distribution")

        # Down Section - Explainability / Preprocessing details
        with gr.Row():
            with gr.Column():
                gr.Markdown("---")
                gr.Markdown("### 🔍 Model Explainability & Preprocessing Diagnostics")
                
                with gr.Tab("1. OpenCV Preprocessing Pipeline"):
                    gr.Markdown(
                        "**Preprocess Diagnostics**: Below are the processing steps showing how "
                        "the raw image is cropped, inverted (white on black background), normalized, and centered."
                    )
                    pipeline_plot = gr.Image(label="Grayscale -> Binary -> Cropped -> Centered (28x28)")
                    
                with gr.Tab("2. Conv Layer 1 Features"):
                    gr.Markdown(
                        "**Feature maps from the first Conv2D layer**: This layer extracts basic, low-level features. "
                        "Notice how filters act as edge detectors (vertical, horizontal, diagonal edges)."
                    )
                    conv1_plot = gr.Plot(label="Conv Layer 1 Activations")
                    
                with gr.Tab("3. Conv Layer 2 Features"):
                    gr.Markdown(
                        "**Feature maps from the second Conv2D layer**: This layer combines edges from the previous layer "
                        "to extract higher-level structures like corners, curves, and loops."
                    )
                    conv2_plot = gr.Plot(label="Conv Layer 2 Activations")

        # Wire Up Event Listeners
        draw_btn.click(
            fn=process_and_predict,
            inputs=draw_input,
            outputs=[output_html, output_chart, pipeline_plot, conv1_plot, conv2_plot]
        )
        
        file_btn.click(
            fn=process_and_predict,
            inputs=file_input,
            outputs=[output_html, output_chart, pipeline_plot, conv1_plot, conv2_plot]
        )
        
    return demo

if __name__ == "__main__":
    demo = build_gradio_app()
    demo.launch()
