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

# -----------------------------
# 1. Parse user input arguments
# -----------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--model', help='Path to YOLO model file (example: "runs/detect/train/weights/best.pt")', required=True)
parser.add_argument('--source', help='Image source, can be image file ("test.jpg"), image folder ("test_dir"), video file ("testvid.mp4"), or index of USB camera ("usb0")', required=True)
parser.add_argument('--thresh', help='Minimum confidence threshold for displaying detected objects (example: "0.4")', default=0.5)
parser.add_argument('--resolution', help='Resolution in WxH to display inference results at (example: "640x480"), otherwise, match source resolution', default=None)
parser.add_argument('--record', help='Record results from video or webcam and save it as "demo1.avi". Must specify --resolution argument to record.', action='store_true')
args = parser.parse_args()

# -----------------------------
# 2. Initialize variables
# -----------------------------
model_path = args.model
img_source = args.source
min_thresh = float(args.thresh)
user_res = args.resolution
record = args.record

# Check if model file exists
if not os.path.exists(model_path):
    print('ERROR: Model path is invalid or model was not found.')
    sys.exit(0)

# Load YOLO model
model = YOLO(model_path, task='detect')
labels = model.names

# -----------------------------
# 3. Determine source type
# -----------------------------
img_ext_list = ['.jpg','.JPG','.jpeg','.JPEG','.png','.PNG','.bmp','.BMP']
vid_ext_list = ['.avi','.mov','.mp4','.mkv','.wmv']

if os.path.isdir(img_source):
    source_type = 'folder'
elif os.path.isfile(img_source):
    _, ext = os.path.splitext(img_source)
    if ext in img_ext_list:
        source_type = 'image'
    elif ext in vid_ext_list:
        source_type = 'video'
    else:
        print(f'File extension {ext} is not supported.')
        sys.exit(0)
elif 'usb' in img_source:
    source_type = 'usb'
    usb_idx = int(img_source[3:])
elif 'picamera' in img_source:
    source_type = 'picamera'
    picam_idx = int(img_source[8:])
else:
    print(f'Input {img_source} is invalid.')
    sys.exit(0)

# -----------------------------
# 4. Parse user-specified resolution
# -----------------------------
resize = False
if user_res:
    resize = True
    resW, resH = map(int, user_res.split('x'))

# -----------------------------
# 5. Recording setup
# -----------------------------
if record:
    if source_type not in ['video','usb']:
        print('Recording only works for video and camera sources.')
        sys.exit(0)
    if not user_res:
        print('Please specify resolution to record video at.')
        sys.exit(0)
    record_name = 'demo1.avi'
    record_fps = 30
    recorder = cv2.VideoWriter(record_name, cv2.VideoWriter_fourcc(*'MJPG'), record_fps, (resW,resH))

# -----------------------------
# 6. Load or initialize image source
# -----------------------------
if source_type == 'image':
    imgs_list = [img_source]
elif source_type == 'folder':
    imgs_list = []
    filelist = glob.glob(img_source + '/*')
    for file in filelist:
        _, file_ext = os.path.splitext(file)
        if file_ext in img_ext_list:
            imgs_list.append(file)
elif source_type == 'video' or source_type == 'usb':
    cap_arg = img_source if source_type == 'video' else usb_idx
    cap = cv2.VideoCapture(cap_arg)
    if user_res:
        cap.set(3, resW)
        cap.set(4, resH)
elif source_type == 'picamera':
    from picamera2 import Picamera2
    cap = Picamera2()
    cap.configure(cap.create_video_configuration(main={"format": 'XRGB8888', "size": (resW, resH)}))
    cap.start()

# -----------------------------
# 7. Bounding box colors
# -----------------------------
bbox_colors = [(164,120,87), (68,148,228), (93,97,209), (178,182,133), (88,159,106),
               (96,202,231), (159,124,168), (169,162,241), (98,118,150), (172,176,184)]

# -----------------------------
# 8. Excel and output folder setup
# -----------------------------
output_folder = "output_images"
os.makedirs(output_folder, exist_ok=True)
excel_file = "detection_results.xlsx"

if os.path.exists(excel_file):
    df = pd.read_excel(excel_file)
else:
    df = pd.DataFrame(columns=["Timestamp", "Image_Name", "Object_Count"])

# -----------------------------
# 9. Initialize variables for FPS
# -----------------------------
avg_frame_rate = 0
frame_rate_buffer = []
fps_avg_len = 200
img_count = 0

# -----------------------------
# 10. Begin inference loop
# -----------------------------
while True:
    t_start = time.perf_counter()

    # Load frame
    if source_type in ['image','folder']:
        if img_count >= len(imgs_list):
            print('All images processed. Exiting.')
            break
        img_filename = imgs_list[img_count]
        frame = cv2.imread(img_filename)
        img_name_only = os.path.basename(img_filename)
        img_count += 1
    elif source_type in ['video','usb']:
        ret, frame = cap.read()
        if not ret:
            print('Reached end of video. Exiting.')
            break
        img_name_only = f"frame_{img_count}.jpg"
        img_count += 1
    elif source_type == 'picamera':
        frame_bgra = cap.capture_array()
        frame = cv2.cvtColor(np.copy(frame_bgra), cv2.COLOR_BGRA2BGR)
        if frame is None:
            print('Unable to read frames from Picamera. Exiting.')
            break
        img_name_only = f"frame_{img_count}.jpg"
        img_count += 1

    # Resize if needed
    if resize:
        frame = cv2.resize(frame,(resW,resH))

    # Run YOLO inference
    results = model(frame, verbose=False)
    detections = results[0].boxes
    object_count = 0

    # Process detections
    for i in range(len(detections)):
        xyxy_tensor = detections[i].xyxy.cpu()
        xyxy = xyxy_tensor.numpy().squeeze()
        xmin, ymin, xmax, ymax = xyxy.astype(int)
        classidx = int(detections[i].cls.item())
        classname = labels[classidx]
        conf = detections[i].conf.item()

        if conf > min_thresh:
            color = bbox_colors[classidx % 10]
            cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), color, 2)
            label = f'{classname}: {int(conf*100)}%'
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_ymin = max(ymin, labelSize[1]+10)
            cv2.rectangle(frame, (xmin,label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), color, cv2.FILLED)
            cv2.putText(frame, label, (xmin,label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0),1)
            object_count += 1

    # Save image with bounding boxes
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    save_name = f"{timestamp_str}_{img_name_only}"
    cv2.imwrite(os.path.join(output_folder, save_name), frame)

    # Save to Excel
    df = pd.concat([df, pd.DataFrame({"Timestamp":[timestamp_str],
                                      "Image_Name":[save_name],
                                      "Object_Count":[object_count]})], ignore_index=True)
    df.to_excel(excel_file, index=False)

    # Display
    cv2.putText(frame, f'Number of objects: {object_count}', (10,40), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,255), 2)
    if source_type in ['video','usb','picamera']:
        cv2.putText(frame, f'FPS: {avg_frame_rate:.2f}', (10,20), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,255), 2)
    cv2.imshow('YOLO detection results', frame)
    if record: recorder.write(frame)

    # Key controls
    if source_type in ['image','folder']:
        key = cv2.waitKey()
    else:
        key = cv2.waitKey(5)

    if key in [ord('q'), ord('Q')]:
        break
    elif key in [ord('s'), ord('S')]:
        cv2.waitKey()
    elif key in [ord('p'), ord('P')]:
        cv2.imwrite('capture.png', frame)

    # Calculate FPS
    t_stop = time.perf_counter()
    frame_rate_calc = 1 / (t_stop - t_start)
    if len(frame_rate_buffer) >= fps_avg_len:
        frame_rate_buffer.pop(0)
    frame_rate_buffer.append(frame_rate_calc)
    avg_frame_rate = np.mean(frame_rate_buffer)

# -----------------------------
# 11. Cleanup
# -----------------------------
print(f'Average pipeline FPS: {avg_frame_rate:.2f}')
if source_type in ['video','usb']: cap.release()
elif source_type == 'picamera': cap.stop()
if record: recorder.release()
cv2.destroyAllWindows()
