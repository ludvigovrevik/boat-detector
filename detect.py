"""
YOLO Detection Script for Images and Videos

Run object detection on images or videos using a trained YOLO model.
"""

import argparse
import os
from pathlib import Path
import cv2
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Run YOLO detection on images or videos")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to trained YOLO model (.pt file)")
    parser.add_argument("--source", type=str, required=True,
                        help="Path to image, video, or directory")
    parser.add_argument("--img-size", type=int, default=1024,
                        help="Inference image size")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45,
                        help="IOU threshold for NMS")
    parser.add_argument("--max-det", type=int, default=300,
                        help="Maximum detections per image")
    parser.add_argument("--output-dir", type=str, default="output",
                        help="Directory to save results")
    parser.add_argument("--save-txt", action="store_true",
                        help="Save results as text files")
    parser.add_argument("--no-save", action="store_true",
                        help="Don't save annotated images/videos")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        return

    if not os.path.exists(args.source):
        print(f"Error: Source not found: {args.source}")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 50)
    print("YOLO Object Detection")
    print("=" * 50)
    print(f"Model:      {args.model}")
    print(f"Source:     {args.source}")
    print(f"Image Size: {args.img_size}")
    print(f"Confidence: {args.conf}")
    print("=" * 50)

    # Check if source is a video
    source_path = Path(args.source)
    is_video = source_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv', '.wmv']

    if is_video:
        cap = cv2.VideoCapture(args.source)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        print(f"\nVideo Info:")
        print(f"  Resolution: {width}x{height}")
        print(f"  FPS: {fps}")
        print(f"  Frames: {frame_count}")
        print(f"  Duration: {frame_count/fps:.1f}s")

    print("\nLoading model...")
    model = YOLO(args.model)

    print("Running detection...")
    results = model.predict(
        source=args.source,
        imgsz=args.img_size,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        save=not args.no_save,
        save_txt=args.save_txt,
        project=args.output_dir,
        name="detect",
        line_width=2,
        show_labels=True,
        show_conf=True,
        verbose=False
    )

    # Print detection statistics
    total_detections = 0
    items_with_detections = 0

    for result in results:
        if result.boxes is not None and len(result.boxes) > 0:
            total_detections += len(result.boxes)
            items_with_detections += 1

    print("\n" + "=" * 50)
    print("Detection Complete!")
    print("=" * 50)
    print(f"Items processed: {len(results)}")
    print(f"Items with detections: {items_with_detections}")
    print(f"Total detections: {total_detections}")

    if not args.no_save:
        output_path = Path(args.output_dir) / "detect"
        print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
