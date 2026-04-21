# importing reqd. libs.

import os
import torch
import numpy as np
import cv2
from data import cfg_mnet
from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
from models.retinaface import RetinaFace
from utils.box_utils import decode

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# params to be configured.

model_path = './weights/mobilenet_0.25_Final.pth'
conf_thresh = 0.81
nms_thresh = 0.4

# load pretrained weights

pretrained_dict = torch.load(model_path, map_location = device)
if 'state_dict' in pretrained_dict:
    pretrained_dict = {
        k.replace('module.', ''): v
        for k, v in pretrained_dict['state_dict'].items()
    }
else:
    pretrained_dict = {
        k.replace('module.', ''): v for k, v in pretrained_dict.items()
    }

model = RetinaFace(cfg = cfg_mnet, phase = 'test')
model.load_state_dict(pretrained_dict, strict = False)
model.eval()
model.to(device)

def generate_retinaface_bboxes(image):
    img_raw = image
    img = np.float32(img_raw)
    h, w, _ = img.shape
    scale = torch.tensor([w, h, w, h], device = device)
    img -= (104, 117, 123)
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0).to(device)

    # inference
    with torch.no_grad():
        loc, conf, _ = model(img)

    # decode boxes
    priorbox = PriorBox(cfg_mnet, image_size = (h, w))
    priors = priorbox.forward().to(device)

    boxes = decode(
        loc.squeeze(0),
        priors.data,
        cfg_mnet['variance']
    )

    boxes = boxes * scale
    boxes = boxes.cpu().numpy()
    scores = conf.squeeze(0).cpu().numpy()[:, 1]

    # filter by confidence
    inds = np.where(scores > conf_thresh)[0]
    boxes = boxes[inds]
    scores = scores[inds]

    # nms
    dets = np.hstack((boxes, scores[:, None])).astype(np.float32)
    keep = py_cpu_nms(dets, nms_thresh)
    dets = dets[keep]

    return dets