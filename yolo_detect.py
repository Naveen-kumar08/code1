import os
import sys
import argparse
import glob
import time

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

# Define and parse user input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', help='Path to YOLO model file (example: "runs/detect/train/weights/best.pt")',
                    required=True)
parser.add_argument('--source', help='Image source, can be image file ("test.jpg"), \
                    image folder ("test_dir"), video file ("testvid.mp4"), or index of USB camera ("usb0")', 
                    required=True)
parser.add_argument('--thresh', help='Minimum confidence threshold for displaying detected objects (example: "0.4")',
                    default=0.5)
parser.add_argument('--resolution', help='Resolution in WxH to display inference results at (example: "640x480"), \
                    otherwise, match source resolution',
                    default=None)
parser.add_argument('--record', help='Record results from video or webcam and save it as "demo1.avi". Must specify --resolution argument to record.',
                    action='store_true')
parser.add_argument('--save_dir', help='Directory to save detected bottle images', default='captures')
parser.add_argument('--log_file', help='Excel file to log detections', default='results.xlsx')

args = parser.parse_args()

# Parse user inputs
model_path = args.model
img_source = args.source
min_thresh = float(args.thresh)
user_res = args.resolution
record = args.record
save_dir = args.save_dir
log_file = args.log_file

# Make save directory
os.makedirs(save_dir, exist_ok=True)

# Initialize logging list
log_data = []

# Check if model file exists
if not os.path.exists(model_path):
    print('ERROR: Model path is invalid or model was not found.')
    sys.exit(0)

# Load YOLO model
model = YOLO(model_path, task='detect')
labels = model.names

# Determine source type
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
        print(f'File extension {ext} not supported.')
        sys.exit(0)
elif 'usb' in img_source:
    source_type = 'usb'
    usb_idx = int(img_source[3:])
else:
    print('Invalid source.')
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
    imgs_list = [f for f in glob.glob(img_source+'/*') if os.path.splitext(f)[1] in img_ext_list]
elif source_type in ['video','usb']:
    cap = cv2.VideoCapture(img_source if source_type=='video' else usb_idx)
    if user_res:
        cap.set(3, resW)
        cap.set(4, resH)

# Colors
bbox_colors = [(164,120,87),(68,148,228),(93,97,209),(178,182,133),(88,159,106),
               (96,202,231),(159,124,168),(169,162,241),(98,118,150),(172,176,184)]

avg_frame_rate = 0
frame_rate_buffer = []
fps_avg_len = 200
img_count = 0

# Main loop
while True:
    t_start = time.perf_counter()

    # Load frame
    if source_type in ['image','folder']:
        if img_count >= len(imgs_list):
            break
        frame = cv2.imread(imgs_list[img_count])
        img_name = os.path.splitext(os.path.basename(imgs_list[img_count]))[0]
        img_count += 1
    else:
        ret, frame = cap.read()
        if not ret:
            break
        img_name = f'frame_{int(time.time()*1000)}'

    if resize:
        frame = cv2.resize(frame, (resW, resH))

    # Run YOLO
    results = model(frame, verbose=False)
    detections = results[0].boxes

    bottle_count = 0
    for det in detections:
        class_idx = int(det.cls.item())
        class_name = labels[class_idx].lower()
        conf = det.conf.item()
        if class_name == 'bottle' and conf > min_thresh:
            xmin, ymin, xmax, ymax = det.xyxy.cpu().numpy().astype(int)[0]
            color = bbox_colors[class_idx % 10]
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
            cv2.putText(frame, f'{class_name}:{int(conf*100)}%',
                        (xmin, ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            bottle_count += 1
            # Save image
            save_path = os.path.join(save_dir, f'{img_name}_{int(time.time()*1000)}.png')
            cv2.imwrite(save_path, frame)
            # Log data
            log_data.append([img_name, class_name, conf, xmin, ymin, xmax, ymax, bottle_count, save_path])

    # FPS calculation
    t_stop = time.perf_counter()
    fps = 1 / max((t_stop - t_start),1e-6)
    frame_rate_buffer.append(fps)
    if len(frame_rate_buffer) > fps_avg_len:
        frame_rate_buffer.pop(0)
    avg_frame_rate = np.mean(frame_rate_buffer)

    cv2.putText(frame, f'FPS: {avg_frame_rate:.2f}', (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
    cv2.putText(frame, f'Bottles: {bottle_count}', (10,45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

    cv2.imshow('YOLO detection results', frame)
    if record:
        recorder.write(frame)

    key = cv2.waitKey(5)
    if key in [ord('q'), ord('Q')]:
        break

# Save Excel log
if log_data:
    df = pd.DataFrame(log_data, columns=["image_name","class","confidence","xmin","ymin","xmax","ymax","object_count","saved_path"])
    df.to_excel(log_file, index=False)

# Cleanup
if source_type in ['video','usb']:
    cap.release()
if record:
    recorder.release()
cv2.destroyAllWindows()
print(f'Average FPS: {avg_frame_rate:.2f}')
