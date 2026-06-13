import argparse
from src.train import train_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the CNN Digit Recognizer model.")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--augment", action="store_true", help="Apply data augmentation during training")
    parser.add_argument("--model_name", type=str, default="digit_recognizer_model.keras", help="Filename for saved model")
    args = parser.parse_args()

    train_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        augment=args.augment,
        model_name=args.model_name
    )
