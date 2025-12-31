import os
import sys
import argparse
import glob
import time
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

# -------------------------
# Arguments
# -------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True, help='Path to YOLO model file (e.g., best.pt)')
parser.add_argument('--source', required=True, help='Image/video source or USB camera (usb0)')
parser.add_argument('--target', default=None, help='Class to detect (optional, e.g., bottle)')
parser.add_argument('--conf', default=0.5, type=float, help='Confidence threshold')
parser.add_argument('--save_dir', default='outputs', help='Directory to save crops and excel')
parser.add_argument('--imgsz', default=640, type=int, help='YOLO input image size')
parser.add_argument('--resolution', default=None, help='Display resolution WxH (e.g., 1280x720)')
args = parser.parse_args()

model_path = args.model
source = args.source
target_class = args.target
conf_thres = args.conf
save_dir = args.save_dir
imgsz = args.imgsz

# Create directories
os.makedirs(save_dir, exist_ok=True)
crop_dir = os.path.join(save_dir, 'crops')
os.makedirs(crop_dir, exist_ok=True)
excel_file = os.path.join(save_dir, 'object_count.xlsx')

# Load previous Excel or create new
if os.path.exists(excel_file):
    df = pd.read_excel(excel_file)
else:
    df = pd.DataFrame(columns=['Filename', 'Class', 'Confidence'])

# -------------------------
# Load YOLO model
# -------------------------
model = YOLO(model_path)
labels = model.names

# -------------------------
# Determine source type
# -------------------------
img_ext_list = ['.jpg','.jpeg','.png','.bmp']
vid_ext_list = ['.avi','.mp4','.mov','.mkv']

if os.path.isdir(source):
    source_type = 'folder'
elif os.path.isfile(source):
    _, ext = os.path.splitext(source)
    if ext.lower() in img_ext_list:
        source_type = 'image'
    elif ext.lower() in vid_ext_list:
        source_type = 'video'
    else:
        print("Unsupported file")
        sys.exit()
elif 'usb' in source:
    source_type = 'usb'
    usb_idx = int(source[3:])
else:
    print("Invalid source")
    sys.exit()

# -------------------------
# Initialize capture
# -------------------------
if source_type in ['video', 'usb']:
    cap = cv2.VideoCapture(usb_idx if source_type=='usb' else source)
    # Optional: Set display resolution if specified
    if args.resolution:
        resW, resH = map(int, args.resolution.lower().split('x'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, resW)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resH)
        resize = True
    else:
        resize = False
else:
    resize = args.resolution is not None
    if resize:
        resW, resH = map(int, args.resolution.lower().split('x'))

# -------------------------
# Inference loop
# -------------------------
img_count = 0
if source_type in ['image','folder']:
    img_files = [f for f in glob.glob(os.path.join(source,'*')) if os.path.splitext(f)[1].lower() in img_ext_list]

while True:
    # Read frame
    if source_type in ['image','folder']:
        if img_count >= len(img_files):
            print("All images processed.")
            break
        frame = cv2.imread(img_files[img_count])
        filename = os.path.basename(img_files[img_count])
        img_count += 1
    else:
        ret, frame = cap.read()
        if not ret:
            print("Video/Camera ended.")
            break
        filename = f"frame_{img_count}.jpg"
        img_count += 1

    # Resize frame for display if requested
    if resize:
        frame = cv2.resize(frame, (resW, resH))

    # YOLO inference
    results = model(frame, imgsz=imgsz, conf=conf_thres, verbose=False)
    detections = results[0].boxes
    object_count = 0

    # Process detections
    for i in range(len(detections)):
        xyxy = detections[i].xyxy.cpu().numpy().astype(int).squeeze()
        cls_id = int(detections[i].cls.cpu().numpy().squeeze())
        conf_score = float(detections[i].conf.cpu().numpy().squeeze())
        class_name = labels[cls_id]

        # Skip if not target class
        if target_class and class_name.lower() != target_class.lower():
            continue
        if conf_score < conf_thres:
            continue

        object_count += 1
        xmin, ymin, xmax, ymax = xyxy

        # Crop detected object
        crop_img = frame[ymin:ymax, xmin:xmax]
        crop_filename = f"{os.path.splitext(filename)[0]}_{class_name}_{object_count}.jpg"
        cv2.imwrite(os.path.join(crop_dir, crop_filename), crop_img)

        # Save info to dataframe
        df = pd.concat([df, pd.DataFrame([{'Filename': crop_filename, 'Class': class_name, 'Confidence': conf_score}])], ignore_index=True)

        # Draw bounding box
        color = (0,255,0)
        cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), color, 2)
        cv2.putText(frame, f"{class_name}:{int(conf_score*100)}%", (xmin, ymin-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display info
    cv2.putText(frame, f"Objects: {object_count}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255),2)
    cv2.imshow("YOLO Detection", frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

# -------------------------
# Cleanup
# -------------------------
if source_type in ['video','usb']:
    cap.release()
cv2.destroyAllWindows()
df.to_excel(excel_file, index=False)
print(f"Saved crops in {crop_dir} and object counts in {excel_file}")
