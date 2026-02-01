"""
Visualization Script

Run model inference and display results interactively.
"""

import argparse
import os
import cv2
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Visualize YOLO predictions")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to YOLO model")
    parser.add_argument("--source", type=str, required=True,
                        help="Path to image or directory")
    parser.add_argument("--img-size", type=int, default=1024,
                        help="Inference image size")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"Error: Model not found: {args.model}")
        return

    if not os.path.exists(args.source):
        print(f"Error: Source not found: {args.source}")
        return

    print("=" * 50)
    print("Prediction Visualization")
    print("=" * 50)
    print(f"Model:  {args.model}")
    print(f"Source: {args.source}")
    print("=" * 50)

    print("\nLoading model...")
    model = YOLO(args.model)

    print("Running inference...")
    results = model.predict(
        source=args.source,
        imgsz=args.img_size,
        conf=args.conf,
        save=False,
        verbose=False
    )

    print(f"\nFound {len(results)} images")
    print("Press any key for next image, 'q' to quit\n")

    for result in results:
        img = result.plot()

        filename = os.path.basename(str(result.path))
        num_boxes = len(result.boxes) if result.boxes is not None else 0
        print(f"{filename}: {num_boxes} detections")

        cv2.namedWindow("Predictions", cv2.WINDOW_NORMAL)
        cv2.imshow("Predictions", img)

        key = cv2.waitKey(0)
        if key == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
