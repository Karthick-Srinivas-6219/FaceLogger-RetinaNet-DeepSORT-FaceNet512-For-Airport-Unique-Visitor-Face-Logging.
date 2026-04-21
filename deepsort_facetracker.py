from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2
import numpy as np
import yaml
import torch


with open('config.yaml' , 'r') as f:
    config =yaml.safe_load(f)['YOLOv5_DeepSORT']['tracker']

DISP_TRACKS = config['disp_tracks']
DISP_OBJ_TRACK_BOX = config['disp_obj_track_box']
OBJ_TRACK_COLOR = tuple(config['obj_track_color'])
OBJ_TRACK_BOX_COLOR = tuple(config['obj_track_box_color'])

# Deep Sort Parameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MAX_AGE = 3               # Maximum number of frames to keep a track alive without new detections. Default is 30

N_INIT = 1                  # Minimum number of detections needed to start a new track. Default is 3

NMS_MAX_OVERLAP = 0.5       # Maximum overlap between bounding boxes allowed for non maximal supression(NMS).
                            #If two bounding boxes overlap by more than this value, the one with the lower confidence score is suppressed. Defaults to 1.0.

MAX_COSINE_DISTANCE = 0.2   # Maximum cosine distance allowed for matching detections to existing tracks. 
                            #If the cosine distance between the detection's feature vector and the track's feature vector is higher than this value, 
                            # the detection is not matched to the track. Defaults to 0.2

NN_BUDGET = None            # Maximum number of features to store in the Nearest Neighbor index. If set to None, the index will have an unlimited budget. 
                            #This parameter affects the memory usage of the tracker. Defaults to None.

OVERRIDE_TRACK_CLASS = None  #Optional override for the Track class used by the tracker. This can be used to subclass the Track class and add custom functionality. Defaults to None.
EMBEDDER = "mobilenet"       #The name of the feature extraction model to use. The options are "mobilenet" or "efficientnet". Defaults to "mobilenet".
HALF = True                  # Whether to use half-precision floating point format for feature extraction. This can reduce memory usage but may result in lower accuracy. Defaults to True
BGR = False                   #Whether to use BGR color format for images. If set to False, RGB format will be used. Defaults to True.
EMBEDDER_GPU = True          #Whether to use GPU for feature extraction. If set to False, CPU will be used. Defaults to True.
EMBEDDER_MODEL_NAME = None   #Optional model name for the feature extraction model. If not provided, the default model for the selected embedder will be used.
EMBEDDER_WTS = None          # Optional path to the weights file for the feature extraction model. If not provided, the default weights for the selected embedder will be used.
POLYGON = False              # Whether to use polygon instead of bounding boxes for tracking. Defaults to False.
TODAY = None                 # Optional argument to set the current date. This is used to calculate the age of each track in days. If not provided, the current date is used.


class DeepSORT_Tracker(): 

    def __init__(self):
        
        self.algo_name ="DeepSORT"
        self.object_tracker = DeepSort(max_age=MAX_AGE,
                n_init=N_INIT,
                nms_max_overlap=NMS_MAX_OVERLAP,
                max_cosine_distance=MAX_COSINE_DISTANCE,
                nn_budget=NN_BUDGET,
                override_track_class=OVERRIDE_TRACK_CLASS,
                embedder=EMBEDDER,
                half=HALF,
                bgr=BGR,
                embedder_gpu=EMBEDDER_GPU,
                embedder_model_name=EMBEDDER_MODEL_NAME,
                embedder_wts=EMBEDDER_WTS,
                polygon=POLYGON,
                today=TODAY)
        
        self.device = device
        
        # Force embedder model to GPU
        if hasattr(self.object_tracker, "embedder") and hasattr(self.object_tracker.embedder, "model"):
            self.object_tracker.embedder.model.to(self.device)
            self.object_tracker.embedder.model.eval()
    
        
    def display_track(self , track_history , tracks_current , img):

        for track in tracks_current:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            
            # Retrieve the current track location(i.e - center of the bounding box) and bounding box
            location = track.to_tlbr()
            bbox = location[:4].astype(int)
            bbox_center = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)

            # Retrieve the previous center location, if available
            prev_centers = track_history.get(track_id ,[])
            prev_centers.append(bbox_center)
            track_history[track_id] = prev_centers

            if DISP_OBJ_TRACK_BOX == True: 
                cv2.rectangle(img,(int(bbox[0]), int(bbox[1])),(int(bbox[2]), int(bbox[3])),(0,0,255),1)
                cv2.putText(img, "ID: " + str(track_id), (int(bbox[0]), int(bbox[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1)

            