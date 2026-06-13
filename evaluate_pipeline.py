import argparse
from src.evaluate import evaluate_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the trained CNN model on MNIST test data.")
    parser.add_argument("--model_name", type=str, default="digit_recognizer_model.keras", help="Model filename in models/ folder")
    args = parser.parse_args()

    evaluate_model(model_name=args.model_name)
