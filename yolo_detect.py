import os
import sys
import time
import argparse
import cv2
import numpy as np
from ultralytics import YOLO

# ==============================
# ARGUMENT PARSER
# ==============================
parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True, help='Path to YOLO model (best.pt)')
parser.add_argument('--source', required=True, help='Image / video file / usb0')
parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold')
parser.add_argument('--imgsz', type=int, default=640, help='YOLO input size')
parser.add_argument('--resolution', default=None, help='Camera resolution WxH (example 1280x720)')
parser.add_argument('--save_dir', default='bbox_crops', help='Bounding box crop save folder')
args = parser.parse_args()

os.makedirs(args.save_dir, exist_ok=True)

# ==============================
# LOAD YOLO MODEL
# ==============================
if not os.path.exists(args.model):
    print("❌ Model file not found")
    sys.exit()

model = YOLO(args.model)

# ==============================
# SOURCE SETUP
# ==============================
if args.source.startswith("usb"):
    cap = cv2.VideoCapture(int(args.source[3:]))

    if args.resolution:
        w, h = map(int, args.resolution.split('x'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

    source_type = "camera"

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

# ==============================
# MAIN LOOP
# ==============================
frame_id = 0

while True:

    if source_type == "image":
        frame = cv2.imread(args.source)
        if frame is None:
            break
    else:
        ret, frame = cap.read()
        if not ret:
            break

    frame_id += 1
    orig_h, orig_w = frame.shape[:2]

    # ------------------------------
    # YOLO INPUT RESIZE ONLY
    # ------------------------------
    yolo_img = cv2.resize(frame, (args.imgsz, args.imgsz))
    scale_x = orig_w / args.imgsz
    scale_y = orig_h / args.imgsz

    # ------------------------------
    # INFERENCE
    # ------------------------------
    results = model(yolo_img, verbose=False)[0]

    if results.boxes is not None:
        for i, box in enumerate(results.boxes):

            conf = float(box.conf[0])
            if conf < args.conf:
                continue

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

            # 🔁 SCALE BACK TO ORIGINAL FRAME
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

            # ------------------------------
            # ✅ CROP ONLY BOUNDING BOX AREA
            # ------------------------------
            crop = frame[y1:y2, x1:x2]
            if crop.size > 0:
                crop_name = f"{args.save_dir}/crop_{frame_id}_{i}.png"
                cv2.imwrite(crop_name, crop)

            # ------------------------------
            # DRAW BOUNDING BOX
            # ------------------------------
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{int(conf*100)}%",
                        (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 0), 2)

    cv2.imshow("YOLO Detection", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

    if source_type == "image":
        cv2.waitKey(0)
        break

# ==============================
# CLEANUP
# ==============================
if source_type != "image":
    cap.release()
cv2.destroyAllWindows()
