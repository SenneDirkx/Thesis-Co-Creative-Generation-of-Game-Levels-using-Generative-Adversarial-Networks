from os import walk
import torch

FOLDER = "./tensorizedConv/"

(_, _, filenames) = walk(FOLDER).__next__()

for filename in filenames:
    
    level = torch.load(FOLDER + filename)
    flipped_x = torch.flip(level, dims=[1])
    flipped_y = torch.flip(level, dims=[2])
    flipped_xy = torch.flip(level, dims=[1, 2])

    if not torch.equal(level, flipped_x):
        torch.save(flipped_x, FOLDER + "flippedx_" + filename)
    
    if not torch.equal(level, flipped_y):
        torch.save(flipped_y, FOLDER + "flippedy_" + filename)
    
    if not torch.equal(level, flipped_xy):
        torch.save(flipped_xy, FOLDER + "flippedxy_" + filename)