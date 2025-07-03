from PIL import Image
import numpy as np
import os, glob

import random

# Data Loader
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as ttf

def get_path(hazy_path, clear_path):
    hazy_path = glob.glob(hazy_path)
    clear_path = glob.glob(clear_path)
    
    hazy_path.sort()
    clear_path.sort()

    # Pengecekan jumlah citra
    assert len(hazy_path) == len(clear_path)
    
    print("==== Check Hazy Images ====")
    for i in hazy_path:
        if not os.path.exists(i):
            print(i)
    
    print("==== Check Clear Images ====")
    for i in clear_path:
        if not os.path.exists(i):
            print(i)
    
    return hazy_path, clear_path


def init_datasets(default_path, path='train', density=['thick', 'moderate', 'thin']):
    hazy_train_datasets, clear_train_datasets = [], []
    
    
    for i in density:
        TRAIN_PATH = f'{i}/{path}'
        hazy_datasets, clear_datasets = get_path(
            hazy_path=os.path.join(default_path, TRAIN_PATH, 'input','*'), 
            clear_path=os.path.join(default_path, TRAIN_PATH, 'target', '*')
        )
        
        if i == 'moderate':
            hazy_datasets = [i for i in hazy_datasets if i.split("/")[-1] not in ['271.png', '265.png']]    # Image corrupted
            clear_datasets = [i for i in clear_datasets if i.split("/")[-1] not in ['271.png', '265.png']]  # Image corrupted
            
        hazy_train_datasets += hazy_datasets
        clear_train_datasets += clear_datasets
        
        hazy_train_datasets.sort()
        clear_train_datasets.sort()

    datasets = np.column_stack((hazy_train_datasets, clear_train_datasets))
    print(datasets.shape)
    return datasets


class DatasetGenerator(Dataset):
    def __init__(self, file_path: list, input_size: tuple=(128, 128), train: bool=True, color_format: str="RGB"):
        super(DatasetGenerator, self).__init__()
        self.data = file_path
        self.train = train
        self.input_size = input_size
        self.color_format = color_format

    def __extract_image(self, path: str, input_size: tuple):
        target = Image.open(path).convert("RGB")
        
        # Convert to tensor & Normalize [0, 1]  
        return ttf.to_tensor(target)

    def __getitem__(self, index):
        # Load image
        input_ = self.__extract_image(self.data[index][0], self.input_size)
        target = self.__extract_image(self.data[index][1], self.input_size)

        # Transform for training
        if self.train:
            aug = random.randint(0, 8)
            
            if aug == 1:
                input_, target = input_.flip(1), target.flip(1)
            elif aug == 2:
                input_, target = input_.flip(2), target.flip(2)
            elif aug == 3:
                input_, target = torch.rot90(input_, dims=(1, 2)), torch.rot90(target, dims=(1, 2))
            elif aug == 4:
                input_, target = torch.rot90(input_, dims=(1, 2), k=2), torch.rot90(target, dims=(1, 2), k=2)
            elif aug == 5:
                input_, target = torch.rot90(input_, dims=(1, 2), k=3), torch.rot90(target, dims=(1, 2), k=3)
            elif aug == 6:
                input_, target = torch.rot90(input_.flip(1), dims=(1, 2)), torch.rot90(target.flip(1), dims=(1, 2))
            elif aug == 7:
                input_, target = torch.rot90(input_.flip(2), dims=(1, 2)), torch.rot90(target.flip(2), dims=(1, 2))
                
        return input_, target

    def __len__(self):
        return len(self.data)