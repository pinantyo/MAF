import json
import os
import numpy as np
import random
import glob
from PIL import Image

import torch
import torchvision.transforms.functional as ttf

import matplotlib.pyplot as plt

def set_seed(seed: int = 42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

def is_image_file(filename: str):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", "JPG"])

def get_directories(path: str):
    files = glob.glob(path)
    files = [i for i in files if is_image_file(i)]
    return files

def train_test_split(input: list, target: list, size: float=0.2):
    # Split Train & Test
    length_data = len(input)
    split = int(length_data * (1-size))

    # Training and Testing
    return [input[:split], target[:split]], [input[split:], target[split:]]


def save_metrics(path, metrics, model, type_data):
    if os.path.exists(path):
        with open(path, "r") as f: 
            global_metrics = json.load(f)
    else:
        global_metrics = {}

    if model not in list(global_metrics.keys()):
        global_metrics[model] = {}

    global_metrics[model][type_data] = metrics
    
    with open(path, "w") as f: 
        json.dump(global_metrics, f)
 
def load_image(path: str, is_normalize: bool=True):
    target = Image.open(path).convert("RGB")
        
    # Convert to tensor & Normalize [0, 1]  

    if is_normalize:
        target = ttf.to_tensor(target)

    return target

def to_categorical(image, semantic_color):
    semantic_map = []

    for colour in semantic_color:
        equality = np.equal(image, colour)
        class_map = np.all(equality, axis = -1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1)
    return semantic_map


def save_image(image, path, save_path, method, is_hazy, mode):
    if is_hazy:
        is_hazy = "hazy"
    else:
        is_hazy = "clear"
        
    file_name = path.split("\\")[-1]

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    save_path = os.path.join(save_path, method)
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    save_path = os.path.join(save_path, is_hazy)
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    save_path = os.path.join(save_path, mode)
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    save_path = os.path.join(save_path, file_name)
    plt.imsave(save_path, image)