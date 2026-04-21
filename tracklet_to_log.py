# loading reqd. libs.

import torch
from torchvision import transforms
from PIL import Image
from facenet_pytorch import InceptionResnetV1
import numpy as np
import os
import shutil

# create gpu instance
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# save params

unique_dir = "unique_faces/"
os.makedirs(unique_dir, exist_ok=True)
unique_embeddings = []
unique_images = []
threshold = 0.74

# load facenet with 'vggface2' pretrained weights (99.65% on LFW)

model = InceptionResnetV1(pretrained = 'vggface2').to(device).eval()

# preprocessing before embedding

embeddings_transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize( # InceptionResNet style
    mean=[0.5, 0.5, 0.5],
    std=[0.5, 0.5, 0.5]
)
])

# load image for inference

def load_img_for_inference(path):
    img = Image.open(path).convert('RGB')
    return embeddings_transform(img)

# function to get facenet embeddings for an image

def get_facenet512_embedding(img):
    img_t = embeddings_transform(img)
    person_crop = img_t.unsqueeze(0).to(device)
    with torch.no_grad():
        get_facenet512_embedding = model(person_crop)
    return get_facenet512_embedding

# function to compute average embedding from tracklet

def get_avg_embedding_from_tracklet(folder_path):
    embeddings = []
    for file_name in os.listdir(folder_path):
        if not file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        img_path = os.path.join(folder_path, file_name)
        img = Image.open(img_path).convert('RGB')
        with torch.no_grad():
            emb = get_facenet512_embedding(img)
            emb = emb.squeeze(0).cpu().numpy()
        embeddings.append(emb)
    if len(embeddings) == 0:
        return None
    avg_embedding = np.mean(embeddings, axis=0)
    return avg_embedding

# helper function to compute euclidean distance

def euclidean_dist(a, b):
    return np.linalg.norm(a-b)

def generate_log_from_tracklets(tracklets_dir):
    for person_folder in sorted(os.listdir(tracklets_dir)):
        folder_path = os.path.join(tracklets_dir, person_folder)

        if not os.path.isdir(folder_path):
            continue
        avg_embedding = get_avg_embedding_from_tracklet(folder_path)
        if avg_embedding is None:
            continue
        is_unique = True
        for stored_emb in unique_embeddings:
            dist = euclidean_dist(avg_embedding, stored_emb)
            if dist < threshold:
                is_unique = False
                break
        if is_unique:
            unique_embeddings.append(avg_embedding)
            for file_name in sorted(os.listdir(folder_path)):
                if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    first_img_path = os.path.join(folder_path, file_name)
                    save_path = os.path.join(unique_dir, f"face_{len(unique_embeddings)}.jpg")
                    shutil.copy(first_img_path, save_path)
                    unique_images.append(save_path)
                    break

