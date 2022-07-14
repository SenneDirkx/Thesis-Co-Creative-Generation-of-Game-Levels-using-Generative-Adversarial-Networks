from copy import deepcopy
import json
import sys
import torch
import torch.nn as nn
import numpy
from torch.autograd import Variable

OUTPUT_FILENAME = sys.argv[1]
DISTANCE_FUNCTION = sys.argv[2]

def dice_score(level1, level2):
    level1_d = get_dummies(level1)
    level2_d = get_dummies(level2)

    a = 0
    b = 0
    #c = 0

    for i in range(len(level1_d)):
        for j in range(len(level1_d[0])):
            for k in range(len(level1_d[0][0])):
                if level1_d[i][j][k] == level2_d[i][j][k] and level1_d[i][j][k] == 1:
                    a += 1
                elif level1_d[i][j][k] != level2_d[i][j][k]:
                    b += 1
    
    result = (2*a)/(2*a + b)
    return result        

def get_dummies(level):
    dummy_level = deepcopy(level)
    for i in range(len(level)):
        for j in range(len(level[0])):
            if level[i][j] == 0:
                dummy_level[i][j] = [1, 0, 0, 0]
            elif level[i][j] == 1:
                dummy_level[i][j] = [0, 1, 0, 0]
            elif level[i][j] == 2:
                dummy_level[i][j] = [0, 0, 1, 0]
            elif level[i][j] == 3:
                dummy_level[i][j] = [0, 0, 0, 1]
            else:
                raise ValueError("MORE THAN 4 CLASSES ???")
    return dummy_level

def main():
    if DISTANCE_FUNCTION == 'dice':
        return main_dice()
    elif DISTANCE_FUNCTION == 'lpips':
        return main_lpips()
    else:
        return None, None

def main_dice():
    levels = []
    with open(OUTPUT_FILENAME, 'r') as level_output:
        for line in level_output:
            list_level = json.loads(line)
            levels.append(list_level)
    
    total_score = 0
    pair_counter = 0
    for i in range(len(levels)):
        for j in range(i+1, len(levels)):
            if i == j:
                continue
            pair_counter += 1
            selected1 = i
            selected2 = j
            score = dice_score(levels[selected1], levels[selected2])
            total_score += score
    avg_score = total_score / pair_counter
    return avg_score, pair_counter

def main_lpips():
    import lpips
    loss_fn_alex = lpips.LPIPS(net='alex', verbose=False, model_path='copypaste/alex.pth')
    upscaler = nn.Upsample(scale_factor=4, mode='nearest')
    levels = []
    with open(OUTPUT_FILENAME, 'r') as level_output:
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
            og_image = image.permute(2,0,1) / 255 * 2 - 1
            batched_image = og_image[None,:]
            upscaled_image = upscaler(batched_image)
            levels.append(upscaled_image)
    total_score = 0
    pair_counter = 0
    MAX_LEVELS = len(levels)
    MAX_LEVELS = 50
    for i in range(MAX_LEVELS):
        for j in range(i+1, MAX_LEVELS):
            if i == j:
                continue
            pair_counter += 1
            selected1 = i
            selected2 = j
            score = loss_fn_alex(levels[selected1], levels[selected2])
            total_score += score
    avg_score = total_score / pair_counter
    return avg_score.item(), pair_counter

score, pairs = main()
print(score, "in", pairs, "pairs")