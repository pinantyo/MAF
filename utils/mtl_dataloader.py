# Data Loader
import random
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset

        
class DatasetGenerator(Dataset):
    def __init__(self, semantic_color, file_path: list, input_size: tuple=(128, 128), train: bool=True):
        super(DatasetGenerator, self).__init__()
        self.data = file_path
        self.train = train
        self.input_size = input_size
        self.semantic_color = semantic_color
        
        
    def __to_categorical(self, image):
        semantic_map = []

        for colour in self.semantic_color:
            equality = np.equal(image, colour)
            class_map = np.all(equality, axis = -1)
            semantic_map.append(class_map)
        semantic_map = np.stack(semantic_map, axis=-1)
        return semantic_map
        
        
    def __transform(self, input_, target, size):
        # Random Crop
        aug = random.randint(0, 8)
            
        if aug == 1:
            input_ = input_.flip(1)
            for i in range(len(target)):
                target[i] = target[i].flip(1)
                
        elif aug == 2:
            input_ = input_.flip(2)
            for i in range(len(target)):
                target[i] = target[i].flip(2)
                
        elif aug == 3:
            input_ = torch.rot90(input_, dims=(1, 2))
            for i in range(len(target)):
                target[i] = torch.rot90(target[i], dims=(1, 2))
                
        elif aug == 4:
            input_ = torch.rot90(input_, dims=(1, 2), k=2)
            for i in range(len(target)):
                target[i] = torch.rot90(target[i], dims=(1, 2), k=2)

        elif aug == 5:
            input_ = torch.rot90(input_, dims=(1, 2), k=3)
            for i in range(len(target)):
                target[i] = torch.rot90(target[i], dims=(1, 2), k=3)
                
        elif aug == 6:
            input_ = torch.rot90(input_.flip(1), dims=(1, 2))
            for i in range(len(target)):
                target[i] = torch.rot90(target[i].flip(1), dims=(1, 2))
                
        elif aug == 7:
            input_ = torch.rot90(input_.flip(2), dims=(1, 2))
            for i in range(len(target)):
                target[i] = torch.rot90(target[i].flip(2), dims=(1, 2))

        return input_, target[0], target[1]
        
    def __load_image(self, path, size):
        image = Image.open(path).convert("RGB")
        return np.asarray(image)

    def __getitem__(self, index):        
        x = self.__load_image(self.data[index][0], self.input_size)
        y_rec = self.__load_image(self.data[index][1], self.input_size)
        y = self.__load_image(self.data[index][2], self.input_size)

        # Penggunaan one hot encoding utk mengubah menjadi (CLASS, H, W)
        y = self.__to_categorical(y)

        # Normalization [0, 1]
        x = torch.from_numpy(x.copy()).type(torch.FloatTensor) / 255.
        y_rec = torch.from_numpy(y_rec.copy()).type(torch.FloatTensor) / 255.
        y = torch.from_numpy(y.copy()).type(torch.FloatTensor)
        
        # Konversi dari (H, W, C) ke (C, H, W) sesuai requirement Conv2 Pytorch
        x = torch.permute(x, (2, 0, 1))
        y_rec = torch.permute(y_rec, (2, 0, 1))
        y = torch.permute(y, (2, 0, 1))
        
        # Augmentasi
        if self.train:
            x, y_rec, y = self.__transform(
                input_=x, 
                target=[y_rec, y], 
                size=self.input_size
            )

        return x, y_rec, y.argmax(dim=0)

    def __len__(self):
        return len(self.data)