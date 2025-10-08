import os
import sys
import argparse
import glob
import time
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

# =======================
# Argument Parser
# =======================
parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True, help='Path to YOLO model file (example: runs/detect/train/weights/best.pt)')
parser.add_argument('--source', required=True, help='Image source: image file, folder, video file, or usb0')
parser.add_argument('--thresh', default=0.5, help='Confidence threshold (default: 0.5)')
parser.add_argument('--resolution', default=None, help='Resolution WxH (example: 640x480)')
parser.add_argument('--record', action='store_true', help='Record output video (works with --resolution)')
args = parser.parse_args()

# =======================
# Basic Setup
# =======================
model_path = args.model
img_source = args.source
min_thresh = float(args.thresh)
user_res = args.resolution
record = args.record

if not os.path.exists(model_path):
    print('ERROR: Model not found. Check the path again.')
    sys.exit(0)

model = YOLO(model_path, task='detect')
labels = model.names

img_ext_list = ['.jpg', '.jpeg', '.png', '.bmp']
vid_ext_list = ['.avi', '.mov', '.mp4', '.mkv', '.wmv']

if os.path.isdir(img_source):
    source_type = 'folder'
elif os.path.isfile(img_source):
    _, ext = os.path.splitext(img_source)
    if ext.lower() in img_ext_list:
        source_type = 'image'
    elif ext.lower() in vid_ext_list:
        source_type = 'video'
    else:
        print('Unsupported file type.')
        sys.exit(0)
elif 'usb' in img_source:
    source_type = 'usb'
    usb_idx = int(img_source[3:])
else:
    print('Invalid source input.')
    sys.exit(0)

resize = False
if user_res:
    resize = True
    resW, resH = map(int, user_res.split('x'))

if record:
    if source_type not in ['video', 'usb']:
        print('Recording only available for video/camera sources.')
        sys.exit(0)
    if not user_res:
        print('Please specify --resolution for recording.')
        sys.exit(0)
    recorder = cv2.VideoWriter('demo1.avi', cv2.VideoWriter_fourcc(*'MJPG'), 30, (resW, resH))

if source_type == 'image':
    imgs_list = [img_source]
elif source_type == 'folder':
    imgs_list = [f for f in glob.glob(img_source + '/*') if os.path.splitext(f)[1].lower() in img_ext_list]
elif source_type in ['video', 'usb']:
    cap = cv2.VideoCapture(img_source if source_type == 'video' else usb_idx)
    if user_res:
        cap.set(3, resW)
        cap.set(4, resH)

bbox_colors = [
    (164, 120, 87), (68, 148, 228), (93, 97, 209), (178, 182, 133),
    (88, 159, 106), (96, 202, 231), (159, 124, 168), (169, 162, 241),
    (98, 118, 150), (172, 176, 184)
]

avg_frame_rate = 0
frame_rate_buffer = []
fps_avg_len = 200
img_count = 0
image_number = 1  # Counter for image naming

# =======================
# Create Folders
# =======================
capture_dir = "captures"
if not os.path.exists(capture_dir):
    os.makedirs(capture_dir)

excel_filename = "results.xlsx"

# =======================
# SMART CAPTURE VARIABLES
# =======================
object_present = False       # Track if bottle currently in frame
last_capture_time = 0        # Track last capture timestamp
capture_delay = 1.5          # Seconds between captures to avoid duplicates

# =======================
# MAIN LOOP
# =======================
while True:
    t_start = time.perf_counter()

    # ----- Load Frame -----
    if source_type in ['image', 'folder']:
        if img_count >= len(imgs_list):
            print("All images processed.")
            break
        frame = cv2.imread(imgs_list[img_count])
        img_count += 1
    else:
        ret, frame = cap.read()
        if not ret:
            print("End of video or camera error.")
            break

    if resize:
        frame = cv2.resize(frame, (resW, resH))

    # ----- YOLO Inference -----
    results = model(frame, verbose=False)
    detections = results[0].boxes
    object_count = 0

    for det in detections:
        xyxy = det.xyxy.cpu().numpy().squeeze().astype(int)
        classidx = int(det.cls.item())
        conf = det.conf.item()
        if conf >= min_thresh:
            xmin, ymin, xmax, ymax = xyxy
            classname = labels[classidx]
            color = bbox_colors[classidx % len(bbox_colors)]
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
            label = f'{classname}: {int(conf * 100)}%'
            cv2.putText(frame, label, (xmin, max(ymin - 10, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            object_count += 1

    # ----- FPS Calculation -----
    t_stop = time.perf_counter()
    frame_rate_calc = float(1 / (t_stop - t_start))
    frame_rate_buffer.append(frame_rate_calc)
    if len(frame_rate_buffer) > fps_avg_len:
        frame_rate_buffer.pop(0)
    avg_frame_rate = np.mean(frame_rate_buffer)

    cv2.putText(frame, f'Objects: {object_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, f'FPS: {avg_frame_rate:.2f}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.imshow("YOLO Detection Results", frame)

    if record:
        recorder.write(frame)

    # =======================
    # SMART CAPTURE LOGIC
    # =======================
    current_time = time.time()
    detected_classes = [labels[int(det.cls.item())] for det in detections if det.conf.item() > min_thresh]

    if "bottle" in detected_classes:  # Change this to your target object name
        if not object_present and (current_time - last_capture_time) > capture_delay:
            object_present = True
            last_capture_time = current_time

            # Save captured frame
            capture_name = f"image{image_number}.jpg"
            capture_path = os.path.join(capture_dir, capture_name)
            cv2.imwrite(capture_path, frame)

            # Save detection info to Excel
            output_data = []
            bottle_conf = [det.conf.item() * 100 for det in detections if labels[int(det.cls.item())] == "bottle"]
            avg_conf = round(np.mean(bottle_conf), 2) if bottle_conf else 0.0

            output_data.append([
                capture_name,
                "bottle",
                avg_conf,
                object_count,
                round(avg_frame_rate, 2),
                capture_name
            ])

            df_new = pd.DataFrame(output_data, columns=['Frame/Image', 'Class', 'Confidence (%)', 'Object Count', 'FPS', 'Captured Image'])

            if not os.path.exists(excel_filename):
                df_new.to_excel(excel_filename, index=False)
            else:
                df_existing = pd.read_excel(excel_filename)
                df_final = pd.concat([df_existing, df_new], ignore_index=True)
                df_final.to_excel(excel_filename, index=False)

            print(f"[INFO] Captured {capture_name}")
            image_number += 1
    else:
        object_present = False  # Reset when bottle leaves view

    # =======================
    # Key Controls
    # =======================
    key = cv2.waitKey(5)
    if key == ord('q') or key == ord('Q'):
        break
    elif key == ord('s') or key == ord('S'):
        cv2.waitKey()
    elif key == ord('p') or key == ord('P'):
        cv2.imwrite('manual_capture.png', frame)

# =======================
# Cleanup
# =======================
print(f"Average pipeline FPS: {avg_frame_rate:.2f}")
if source_type in ['video', 'usb']:
    cap.release()
if record:
    recorder.release()
cv2.destroyAllWindows()
