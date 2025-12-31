import os
import sys
import time
import argparse
import cv2
import numpy as np
from ultralytics import YOLO

# -----------------------------
# ARGUMENTS
# -----------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True, help='Path to YOLO model (best.pt)')
parser.add_argument('--source', required=True, help='Image / video / usb0')
parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold')
parser.add_argument('--imgsz', type=int, default=640, help='YOLO resize size (square)')
parser.add_argument('--save_dir', default='impurity_crops', help='Crop save folder')
args = parser.parse_args()

os.makedirs(args.save_dir, exist_ok=True)

# -----------------------------
# LOAD MODEL
# -----------------------------
if not os.path.exists(args.model):
    print("❌ Model not found")
    sys.exit()

model = YOLO(args.model)

# -----------------------------
# SOURCE SETUP
# -----------------------------
if args.source.startswith("usb"):
    cap = cv2.VideoCapture(int(args.source[3:]))
    source_type = "cam"
elif os.path.isfile(args.source):
    ext = os.path.splitext(args.source)[1].lower()
    if ext in [".jpg", ".jpeg", ".png"]:
        source_type = "image"
    else:
        cap = cv2.VideoCapture(args.source)
        source_type = "video"
else:
    print("❌ Invalid source")
    sys.exit()

# -----------------------------
# MAIN LOOP
# -----------------------------
while True:

    if source_type == "image":
        orig_frame = cv2.imread(args.source)
        if orig_frame is None:
            break
    else:
        ret, orig_frame = cap.read()
        if not ret:
            break

    orig_h, orig_w = orig_frame.shape[:2]

    # -----------------------------
    # RESIZE FOR YOLO
    # -----------------------------
    resized_frame = cv2.resize(orig_frame, (args.imgsz, args.imgsz))

    scale_x = orig_w / args.imgsz
    scale_y = orig_h / args.imgsz

    # -----------------------------
    # YOLO INFERENCE
    # -----------------------------
    results = model(resized_frame, verbose=False)[0]

    if results.boxes is not None:
        for i, box in enumerate(results.boxes):

            conf = float(box.conf[0])
            if conf < args.conf:
                continue

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

            # -----------------------------
            # SCALE BACK TO ORIGINAL FRAME
            # -----------------------------
            x1 = int(x1 * scale_x)
            y1 = int(y1 * scale_y)
            x2 = int(x2 * scale_x)
            y2 = int(y2 * scale_y)

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(orig_w, x2)
            y2 = min(orig_h, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            # -----------------------------
            # CROP ONLY IMPURITY REGION
            # -----------------------------
            crop = orig_frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            fname = f"{args.save_dir}/impurity_{int(time.time())}_{i}.png"
            cv2.imwrite(fname, crop)

            # Draw bbox (for visual)
            cv2.rectangle(orig_frame, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(orig_frame, f"{int(conf*100)}%",
                        (x1, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    cv2.imshow("YOLO Detection", orig_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if source_type == "image":
        cv2.waitKey(0)
        break

# -----------------------------
# CLEANUP
# -----------------------------
if source_type != "image":
    cap.release()
cv2.destroyAllWindows()
