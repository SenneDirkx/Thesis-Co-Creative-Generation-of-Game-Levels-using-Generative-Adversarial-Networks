import json
import numpy as np
import imageio
import torch
from torchvision.utils import make_grid
import torchvision.transforms as transforms
import imageio
import torch.nn as nn
to_pil_image = transforms.ToPILImage()
import sys

upscaler = nn.Upsample(scale_factor=10, mode='nearest')

CP = False

L = []
LR = []
K = []
for i in range(3):
    L.append(sys.argv[3*i+1])
    LR.append(sys.argv[3*i+2])
    K.append(sys.argv[3*i+3])

SETUP = sys.argv[10]
N = 5000

for i in range(3):
    
    if CP:
        file_input = f"./copypaste/eval_output/setup{SETUP}/rewritten_output_c{K[i]}_n{N}_l{L[i]}_d1_lr{LR[i]}.json"
        file_output = f"../../../paper/latex/figures/scenarios/cp_scen{SETUP}_selected{i+1}_{K[i]}_{L[i]}_{LR[i].split('.')[1]}.png"
        file_pdf = f"../../../paper/latex/figures/scenarios/cp_scen{SETUP}_selected{i+1}_{K[i]}_{L[i]}_{LR[i].split('.')[1]}.pdf"
    else:
        file_input = f"./eval_output_custom_local/setup{SETUP}/rewritten_output_c{K[i]}_v{N}_n{N}_l{L[i]}_d1_lr{LR[i]}.json"
        file_output = f"../../../paper/latex/figures/scenarios/custom_scen{SETUP}_selected{i+1}_{K[i]}_{L[i]}_{LR[i].split('.')[1]}.png"
        file_pdf = f"../../../paper/latex/figures/scenarios/custom_scen{SETUP}_selected{i+1}_{K[i]}_{L[i]}_{LR[i].split('.')[1]}.pdf"
    counter = 0
    num_lines = sum(1 for _ in open(file_input))
    images = torch.zeros((num_lines, 3, 16, 11), dtype=torch.uint8)
    with open(file_input, 'r') as level_output:
        
        for line in level_output:
            list_level = json.loads(line)
            height = len(list_level)
            width = len(list_level[0])
            image = torch.zeros((height, width, 3), dtype=torch.uint8)
            for y in range(height):
                for x in range(width):
                    tile = list_level[y][x]
                    if tile == 0:
                        image[y,x] = torch.tensor([178,216,216], dtype=torch.uint8)
                    elif tile == 1:
                        image[y,x] = torch.tensor((0,128,128), dtype=torch.uint8)
                    elif tile == 2:
                        image[y,x] = torch.tensor((17, 48, 143), dtype=torch.uint8)
                    elif tile == 3:
                        image[y,x] = torch.tensor((74, 41, 23), dtype=torch.uint8)
                    else:
                        image[y,x] = torch.tensor((255,255,255), dtype=torch.uint8)
            images[counter] = image.permute(2,0,1)
            counter += 1
        
    images = make_grid(images, pad_value=255.0)
    batched_image = images[None,:]
    upscaled_image = upscaler(batched_image)[0]
    upscaled_image = np.array(to_pil_image(upscaled_image))
    imageio.imwrite(file_output, upscaled_image)