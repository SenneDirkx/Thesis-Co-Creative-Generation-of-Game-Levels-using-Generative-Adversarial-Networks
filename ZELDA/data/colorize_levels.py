import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torchvision.transforms as transforms
from torchvision.utils import make_grid
import imageio

class ZeldaLevelDataset(Dataset):
    """Zelda Game levels dataset."""

    def __init__(self, data_dir):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data_dir = data_dir
        self.levels = self.load_levels()

    def __len__(self):
        return len(self.levels)

    def __getitem__(self, idx):
        return self.levels[idx]
    
    def load_levels(self):
        levels = []
        directory = os.fsencode(self.data_dir)
        for file in os.listdir(directory):
            filename = os.fsdecode(file)
            if filename.endswith(".pt"): 
                level = torch.load(self.data_dir + '/' + filename).float()
                levels.append(level)
        return levels

def transform_to_image_format(level):
    colorMap = torch.tensor([[0.11,0.48,0.48,0.05,0.05,0.05,0.29,0.48,0.48,0],
                             [0.75,0.2, 0.2, 0.15,0.15,0.15,0.16,0.2, 0.2, 0],
                             [0.54,0.93,0.93,0.58,0.68,0.68,0.09,0.93,0.93,0]]).transpose(0,1)
    permuted_level = level.permute(0,2,3,1)
    
    colored_permuted_level = torch.matmul(permuted_level,colorMap)
    colored_level = colored_permuted_level.permute(0,3,1,2)
    
    return colored_level

to_pil_image = transforms.ToPILImage()

train_data = ZeldaLevelDataset('./tensorizedConv/')
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

levels = iter(train_loader).next()
print(levels.shape)
levels = transform_to_image_format(levels)
print(levels.shape)
levels = make_grid(levels)
print(levels.shape)
levels = to_pil_image(levels)
print(levels)
levels.save('colorized_levels.png')