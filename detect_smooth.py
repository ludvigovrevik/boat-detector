"""
Smooth Video Detection Script

Run YOLO detection on video with temporal smoothing to reduce flickering
and fill in missing detections.
"""

import argparse
import os
import cv2
import numpy as np
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Run smooth YOLO detection on video")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to trained YOLO model (.pt file)")
    parser.add_argument("--video", type=str, required=True,
                        help="Path to input video")
    parser.add_argument("--output", type=str, default=None,
                        help="Output video path (default: <input>_detected.mp4)")
    parser.add_argument("--img-size", type=int, default=1024,
                        help="Inference image size")
    parser.add_argument("--conf", type=float, default=0.15,
                        help="Confidence threshold (lower for video)")
    parser.add_argument("--max-seconds", type=int, default=None,
                        help="Maximum seconds to process (default: full video)")
    parser.add_argument("--smooth-window", type=int, default=2,
                        help="Temporal smoothing window size")
    parser.add_argument("--box-color", type=str, default="yellow",
                        choices=["yellow", "green", "red", "blue"],
                        help="Bounding box color")
    parser.add_argument("--show-labels", action="store_true",
                        help="Show class labels on boxes")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        return

    if not os.path.exists(args.video):
        print(f"Error: Video file not found: {args.video}")
        return

    # Set output path
    if args.output is None:
        base = os.path.splitext(args.video)[0]
        args.output = f"{base}_detected.mp4"

    # Color mapping (BGR format)
    colors = {
        "yellow": (0, 255, 255),
        "green": (0, 255, 0),
        "red": (0, 0, 255),
        "blue": (255, 0, 0)
    }
    box_color = colors[args.box_color]

    print("=" * 50)
    print("Smooth Video Detection")
    print("=" * 50)
    print(f"Model:  {args.model}")
    print(f"Input:  {args.video}")
    print(f"Output: {args.output}")
    print("=" * 50)

    # Load model
    print("\nLoading model...")
    model = YOLO(args.model)

    # Open video
    cap = cv2.VideoCapture(args.video)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Calculate frames to process
    if args.max_seconds:
        frames_to_process = min(int(fps * args.max_seconds), total_frames)
    else:
        frames_to_process = total_frames

    print(f"\nVideo Info:")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps}")
    print(f"  Total Frames: {total_frames}")
    print(f"  Processing: {frames_to_process} frames")

    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    # Pass 1: Detect objects in all frames
    print("\nPass 1: Detecting objects...")
    detections = []
    frames = []

    for frame_idx in range(frames_to_process):
        ret, frame = cap.read()
        if not ret:
            break

        frames.append(frame)

        results = model.predict(
            source=frame,
            imgsz=args.img_size,
            conf=args.conf,
            iou=0.45,
            max_det=10,
            verbose=False
        )

        if results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            best_idx = boxes.conf.argmax()
            box_xyxy = boxes.xyxy[best_idx].cpu().numpy()
            confidence = boxes.conf[best_idx].cpu().numpy()
            class_idx = int(boxes.cls[best_idx].item())
            class_name = results[0].names[class_idx]
            detections.append({
                'box': box_xyxy,
                'conf': confidence,
                'class': class_name
            })
        else:
            detections.append(None)

        if frame_idx % 30 == 0:
            print(f"  Frame {frame_idx}/{frames_to_process}")

    cap.release()

    # Pass 2: Fill missing detections
    print("\nPass 2: Filling gaps...")
    smoothed = []
    last_valid = None

    for det in detections:
        if det is not None:
            last_valid = det
            smoothed.append(det)
        elif last_valid is not None:
            smoothed.append(last_valid)
        else:
            smoothed.append(None)

    # Pass 3: Temporal smoothing
    print("\nPass 3: Smoothing...")
    window = args.smooth_window
    final = []

    for i in range(len(smoothed)):
        if smoothed[i] is not None and detections[i] is not None:
            boxes_to_avg = []
            weights = []

            for j in range(max(0, i - window), min(len(detections), i + window + 1)):
                if detections[j] is not None:
                    dist = abs(j - i)
                    weight = 1.0 / (1.0 + dist * 0.5)
                    boxes_to_avg.append(detections[j]['box'])
                    weights.append(weight)

            if boxes_to_avg:
                weights = np.array(weights) / np.sum(weights)
                avg_box = np.average(boxes_to_avg, axis=0, weights=weights)
                final.append({'box': avg_box, 'class': smoothed[i]['class']})
            else:
                final.append(smoothed[i])
        elif smoothed[i] is not None:
            final.append(smoothed[i])
        else:
            final.append(None)

    # Pass 4: Write output video
    print("\nPass 4: Writing video...")
    for frame_idx in range(frames_to_process):
        frame = frames[frame_idx].copy()

        if final[frame_idx] is not None:
            box = final[frame_idx]['box']
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 3)

            if args.show_labels:
                label = final[frame_idx].get('class', 'object')
                cv2.putText(frame, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, box_color, 2)

        out.write(frame)

        if frame_idx % 30 == 0:
            print(f"  Frame {frame_idx}/{frames_to_process}")

    out.release()

    # Statistics
    original_detections = sum(1 for d in detections if d is not None)
    final_detections = sum(1 for d in final if d is not None)

    print("\n" + "=" * 50)
    print("Processing Complete!")
    print("=" * 50)
    print(f"Output: {args.output}")
    print(f"\nStatistics:")
    print(f"  Original detections: {original_detections}/{frames_to_process}")
    print(f"  Final detections: {final_detections}/{frames_to_process}")
    print(f"  Gaps filled: {final_detections - original_detections}")


if __name__ == "__main__":
    main()
