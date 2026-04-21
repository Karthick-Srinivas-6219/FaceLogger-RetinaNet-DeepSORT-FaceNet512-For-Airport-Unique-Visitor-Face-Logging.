import cv2
import os
import time
import pandas as pd
from datetime import datetime


# helper function for saving face trackets as folders
def save_face_tracklets(frame, track, base_dir = 'face_tracklets/', save_every = 20, blur_thresh = 100):
    track_id = track.track_id
    timestamp = time.time()
    time_str = datetime.fromtimestamp(timestamp).strftime("%d%m%Y_%H%M%S")
    img_name = f"{track_id}_{time_str}.jpg"
    timestamp = time_str

    if not hasattr(track, 'save_counter'): # counts how many times a track has been saved 
        track.save_counter = 0
    track.save_counter += 1
    if track.save_counter % save_every != 0:
        return None, None, None,"N" # 1 in 3
    
    # get bbox coords
    l, t, r, b = track.to_ltrb()
    l, t, r, b = map(int, [l, t, r, b])

    # frame boundary check
    h_frame, w_frame = frame.shape[:2]
    l = max(0, l)
    t = max(0, t)
    r = min(w_frame, r)
    b = min(h_frame, b)

    crop = frame[t:b, l:r]
    if crop.size == 0:
        return None, None, None,"N" 
    
    # blur detection by laplacian invariance
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()

    if blur_score < blur_thresh:
        return None, None, None,"N"
    
    
    if not track.is_confirmed():
        return img_name, timestamp, track_id,"D" # save only confirmed tracks to avoid noise
    else:
    # create a folder for the person and save the bbox crops
        face_dir = os.path.join(base_dir, f"face_{track_id}")
        os.makedirs(face_dir, exist_ok=True)
        # save image
        cv2.imwrite(os.path.join(face_dir, img_name), crop)
        return img_name, timestamp, track_id,"Y"

def generate_csv_from_folder(folder_path, output_csv="unique_faces_report.csv"):
    image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
    images = [
        f for f in os.listdir(folder_path)
        if f.lower().endswith(image_extensions)
    ]
    images.sort()
    ids = list(range(1, len(images) + 1))
    df = pd.DataFrame({
        "Image_Name": images,
        "ID": ids
    })
    summary_row = pd.DataFrame({
        "Image_Name": ["Total no. of unique individuals"],
        "ID": [len(images)]
    })
    df = pd.concat([df, summary_row], ignore_index=True)
    df.to_csv(output_csv, index=False)
    print(f"CSV file saved as: {output_csv}")