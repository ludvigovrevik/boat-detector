"""
YOLO Training Script for Boat Detection

Train a YOLO model on a custom boat detection dataset.
"""

import argparse
import os
import torch
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Train YOLO model for boat detection")
    parser.add_argument("--model", type=str, default="yolo11n.pt",
                        help="YOLO model to use (e.g., yolo11n.pt, yolo11s.pt, yolo11m.pt)")
    parser.add_argument("--data", type=str, default="data/dataset.yaml",
                        help="Path to dataset YAML file")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size for training")
    parser.add_argument("--img-size", type=int, default=1024,
                        help="Image size for training")
    parser.add_argument("--patience", type=int, default=10,
                        help="Early stopping patience")
    parser.add_argument("--output-dir", type=str, default="output",
                        help="Directory to save outputs")
    parser.add_argument("--name", type=str, default="train",
                        help="Name for this training run")
    args = parser.parse_args()

    torch.cuda.empty_cache()
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 50)
    print("YOLO Boat Detection Training")
    print("=" * 50)
    print(f"Model:      {args.model}")
    print(f"Dataset:    {args.data}")
    print(f"Epochs:     {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Image Size: {args.img_size}")
    print(f"Patience:   {args.patience}")
    print("=" * 50)

    model = YOLO(args.model)

    print("\nStarting training...")
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.img_size,
        batch=args.batch_size,
        name=args.name,
        plots=True,
        amp=True,
        patience=args.patience,
        cos_lr=True,
        workers=0  # Set to 0 to avoid multiprocessing issues on Windows
    )

    print("\nEvaluating on validation set...")
    val_metrics = model.val(data=args.data, split="val", imgsz=args.img_size)

    print("\n" + "=" * 50)
    print("Validation Results:")
    print("=" * 50)
    if hasattr(val_metrics, "box"):
        if hasattr(val_metrics.box, "map50"):
            print(f"mAP50:    {float(val_metrics.box.map50):.4f}")
        if hasattr(val_metrics.box, "map"):
            print(f"mAP50-95: {float(val_metrics.box.map):.4f}")

    # Save model
    model_path = os.path.join(args.output_dir, f"best_{args.name}.pt")
    model.save(model_path)
    print(f"\nModel saved to: {model_path}")
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
