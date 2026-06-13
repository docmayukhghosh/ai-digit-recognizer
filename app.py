import os
from src.app import build_gradio_app

# Build the Gradio Blocks application
app = build_gradio_app()

# Hugging Face Spaces expects the app to be run or imported as "demo" or launched directly.
# Defining "demo" is a standard convention.
demo = app

if __name__ == "__main__":
    app.launch()
