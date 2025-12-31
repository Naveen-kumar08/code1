import os
import sys
import argparse
import glob
import time

import cv2
import numpy as np
from ultralytics import YOLO
import pandas as pd

# Define and parse user input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', help='Path to YOLO model file', required=True)
parser.add_argument('--source', help='Image/video folder/file or USB camera ("usb0")', required=True)
parser.add_argument('--thresh', help='Confidence threshold for detections', default=0.5)
parser.add_argument('--resolution', help='Display resolution WxH', default=None)
parser.add_argument('--record', help='Record video output as demo1.avi', action='store_true')
args = parser.parse_args()

# Parse user inputs
model_path = args.model
img_source = args.source
min_thresh = float(args.thresh)
user_res = args.resolution
record = args.record

# Check if model file exists
if not os.path.exists(model_path):
    print('ERROR: Model path is invalid')
    sys.exit(0)

# Load YOLO model
model = YOLO(model_path, task='detect')
labels = model.names

# Create output folder for captures
os.makedirs("captures", exist_ok=True)
log_data = []

# Determine source type
img_ext_list = ['.jpg','.JPG','.jpeg','.JPEG','.png','.PNG','.bmp','.BMP']
vid_ext_list = ['.avi','.mov','.mp4','.mkv','.wmv']

if os.path.isdir(img_source):
    source_type = 'folder'
elif os.path.isfile(img_source):
    _, ext = os.path.splitext(img_source)
    if ext.lower() in img_ext_list:
        source_type = 'image'
    elif ext.lower() in vid_ext_list:
        source_type = 'video'
    else:
        print(f'File extension {ext} not supported')
        sys.exit(0)
elif 'usb' in img_source:
    source_type = 'usb'
    usb_idx = int(img_source[3:])
else:
    print('Invalid source')
    sys.exit(0)

# Resolution
resize = False
if user_res:
    resize = True
    resW, resH = map(int, user_res.split('x'))

# Recording
if record:
    if source_type not in ['video','usb'] or not user_res:
        print('Recording requires video/camera + resolution')
        sys.exit(0)
    recorder = cv2.VideoWriter('demo1.avi', cv2.VideoWriter_fourcc(*'MJPG'), 30, (resW,resH))

# Load source
if source_type == 'image':
    imgs_list = [img_source]
elif source_type == 'folder':
    imgs_list = [f for f in glob.glob(img_source+'/*') if os.path.splitext(f)[1].lower() in img_ext_list]
elif source_type in ['video','usb']:
    cap = cv2.VideoCapture(img_source if source_type=='video' else usb_idx)
    if user_res:
        cap.set(3, resW)
        cap.set(4, resH)

# Choose your bounding box color for bottles (BGR)
bottle_color = (0, 0, 255)  # Red color for bottle bounding boxes

# FPS tracking
avg_frame_rate = 0
frame_rate_buffer = []
fps_avg_len = 200
img_count = 0
frame_idx = 0

# Inference loop
while True:
    t_start = time.perf_counter()

    if source_type in ['image','folder']:
        if img_count >= len(imgs_list):
            print('All images processed.')
            break
        frame = cv2.imread(imgs_list[img_count])
        img_count += 1
    else:
        ret, frame = cap.read()
        if not ret:
            print('Video ended or camera disconnected.')
            break

    if resize:
        frame = cv2.resize(frame,(resW,resH))

    results = model(frame, verbose=False)
    detections = results[0].boxes
    bottle_count = 0
    saved_frame_name = None

    for i, det in enumerate(detections):
        conf = det.conf.item()
        classidx = int(det.cls.item())
        classname = labels[classidx]

        # Only process bottles above confidence threshold
        if classname.lower() == 'bottle' and conf > min_thresh:
            xmin, ymin, xmax, ymax = det.xyxy.cpu().numpy().astype(int)[0]

            # Draw bounding box
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), bottle_color, 2)
            cv2.putText(frame, f'{classname}: {int(conf*100)}%',
                        (xmin, ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, bottle_color, 2)

            bottle_count += 1

    # Save frame if at least one bottle detected
    if bottle_count > 0:
        timestamp = int(time.time()*1000)
        saved_frame_name = f'captures/frame_{frame_idx}_{timestamp}.png'
        cv2.imwrite(saved_frame_name, frame)

    # Log info
    log_data.append({
        "timestamp": int(time.time()),
        "frame_name": saved_frame_name,
        "bottle_count": bottle_count,
        "detections": [
            {
                "class": labels[int(det.cls.item())],
                "confidence": det.conf.item(),
                "xmin": int(det.xyxy.cpu().numpy()[0][0]),
                "ymin": int(det.xyxy.cpu().numpy()[0][1]),
                "xmax": int(det.xyxy.cpu().numpy()[0][2]),
                "ymax": int(det.xyxy.cpu().numpy()[0][3])
            }
            for det in detections
            if labels[int(det.cls.item())].lower() == 'bottle' and det.conf.item() > min_thresh
        ]
    })

    frame_idx += 1

    # FPS calculation
    t_stop = time.perf_counter()
    fps = 1 / max((t_stop - t_start),1e-6)
    frame_rate_buffer.append(fps)
    if len(frame_rate_buffer) > fps_avg_len:
        frame_rate_buffer.pop(0)
    avg_frame_rate = np.mean(frame_rate_buffer)

    cv2.putText(frame,f'FPS:{avg_frame_rate:.2f}',(10,20),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2)
    cv2.putText(frame,f'Bottles:{bottle_count}',(10,45),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2)

    cv2.imshow('YOLO detection results', frame)
    if record:
        recorder.write(frame)

    if cv2.waitKey(1) & 0xFF in [ord('q'), ord('Q')]:
        break

# Save log to Excel
if log_data:
    # Flatten data for Excel: one row per detected bottle
    rows = []
    for entry in log_data:
        frame_name = entry["frame_name"]
        for det in entry["detections"]:
            rows.append({
                "timestamp": entry["timestamp"],
                "frame_name": frame_name,
                "bottle_count": entry["bottle_count"],
                "class": det["class"],
                "confidence": det["confidence"],
                "xmin": det["xmin"],
                "ymin": det["ymin"],
                "xmax": det["xmax"],
                "ymax": det["ymax"]
            })
    df = pd.DataFrame(rows)
    df.to_excel("results.xlsx", index=False)

# Cleanup
print(f'Average FPS: {avg_frame_rate:.2f}')
if source_type in ['video','usb']:
    cap.release()
if record:
    recorder.release()
cv2.destroyAllWindows()
