import os
from src.app import build_gradio_app

if __name__ == "__main__":
    model_path = os.path.join("models", "digit_recognizer_model.keras")
    if not os.path.exists(model_path):
        print("=" * 60)
        print("WARNING: Trained model not found in models/digit_recognizer_model.keras!")
        print("Please run 'python train_pipeline.py' first to train the model,")
        print("or place a pre-trained Keras model in the models/ directory.")
        print("=" * 60)
        
    app = build_gradio_app()
    # Launch on 0.0.0.0 port 7860 with public sharing enabled
    app.launch(server_name="0.0.0.0", server_port=7860, share=True)
