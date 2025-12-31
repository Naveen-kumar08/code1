import os
import sys
import argparse
import glob
import time
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

# =========================
# ARGUMENT PARSING
# =========================
parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True)
parser.add_argument('--source', required=True)
parser.add_argument('--thresh', default=0.5)
parser.add_argument('--resolution', default=None)
parser.add_argument('--record', action='store_true')

args = parser.parse_args()

model_path = args.model
img_source = args.source
min_thresh = float(args.thresh)
user_res = args.resolution
record = args.record

# =========================
# CHECK MODEL
# =========================
if not os.path.exists(model_path):
    print("Model not found")
    sys.exit(0)

# =========================
# LOAD MODEL
# =========================
model = YOLO(model_path)
labels = model.names

# =========================
# OUTPUT SETUP
# =========================
os.makedirs("bbox_outputs", exist_ok=True)
excel_file = "detection_log.xlsx"

if not os.path.exists(excel_file):
    df = pd.DataFrame(columns=[
        "DateTime",
        "Source",
        "Class",
        "Confidence",
        "Image_Name",
        "Object_Count"
    ])
    df.to_excel(excel_file, index=False)

# =========================
# SOURCE TYPE CHECK
# =========================
img_ext = ['.jpg','.jpeg','.png','.bmp']
vid_ext = ['.avi','.mp4','.mkv','.mov']

if os.path.isdir(img_source):
    source_type = 'folder'
elif os.path.isfile(img_source):
    _, ext = os.path.splitext(img_source)
    if ext.lower() in img_ext:
        source_type = 'image'
    elif ext.lower() in vid_ext:
        source_type = 'video'
    else:
        print("Unsupported file")
        sys.exit(0)
elif 'usb' in img_source:
    source_type = 'usb'
    usb_idx = int(img_source[3:])
else:
    print("Invalid source")
    sys.exit(0)

# =========================
# RESOLUTION
# =========================
resize = False
if user_res:
    resize = True
    resW, resH = map(int, user_res.split('x'))

# =========================
# VIDEO SETUP
# =========================
if source_type in ['video','usb']:
    cap = cv2.VideoCapture(img_source if source_type=='video' else usb_idx)
    if resize:
        cap.set(3, resW)
        cap.set(4, resH)

# =========================
# IMAGE LIST
# =========================
if source_type == 'image':
    imgs = [img_source]
elif source_type == 'folder':
    imgs = [f for f in glob.glob(img_source+"/*") if os.path.splitext(f)[1].lower() in img_ext]

# =========================
# COLORS
# =========================
bbox_colors = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(255,0,255)]

# =========================
# LOOP
# =========================
img_count = 0
fps_buffer = []

while True:

    start = time.perf_counter()

    if source_type in ['image','folder']:
        if img_count >= len(imgs):
            break
        frame = cv2.imread(imgs[img_count])
        img_count += 1
    else:
        ret, frame = cap.read()
        if not ret:
            break

    if resize:
        frame = cv2.resize(frame,(resW,resH))

    results = model(frame, verbose=False)
    detections = results[0].boxes
    object_count = 0

    for det in detections:
        conf = det.conf.item()
        if conf < min_thresh:
            continue

        xmin, ymin, xmax, ymax = det.xyxy.cpu().numpy().squeeze().astype(int)
        class_id = int(det.cls.item())
        classname = labels[class_id]

        color = bbox_colors[class_id % len(bbox_colors)]
        cv2.rectangle(frame,(xmin,ymin),(xmax,ymax),color,2)

        label = f"{classname} {conf:.2f}"
        cv2.putText(frame,label,(xmin,ymin-10),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,color,2)

        # =========================
        # CROP & SAVE BBOX
        # =========================
        crop = frame[ymin:ymax, xmin:xmax]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        crop_name = f"{classname}_{timestamp}.png"

        if crop.size > 0:
            cv2.imwrite(os.path.join("bbox_outputs", crop_name), crop)

        # =========================
        # EXCEL UPDATE (REPEAT OK)
        # =========================
        df = pd.read_excel(excel_file)
        new_row = {
            "DateTime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Source": img_source,
            "Class": classname,
            "Confidence": round(conf,2),
            "Image_Name": crop_name,
            "Object_Count": object_count + 1
        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_excel(excel_file, index=False)

        object_count += 1

    # =========================
    # DISPLAY
    # =========================
    cv2.putText(frame,f"Objects: {object_count}",(10,30),
                cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,255),2)

    cv2.imshow("YOLO Detection", frame)

    key = cv2.waitKey(0 if source_type in ['image','folder'] else 5)
    if key in [ord('q'),ord('Q')]:
        break

    stop = time.perf_counter()
    fps_buffer.append(1/(stop-start))

# =========================
# CLEANUP
# =========================
if source_type in ['video','usb']:
    cap.release()

cv2.destroyAllWindows()

print(f"Average FPS: {np.mean(fps_buffer):.2f}")
