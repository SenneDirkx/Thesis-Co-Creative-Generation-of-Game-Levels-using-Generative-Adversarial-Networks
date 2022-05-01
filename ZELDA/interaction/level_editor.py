import torch
import numpy as np
from torchvision.utils import make_grid
import torchvision.transforms as transforms
from output_utils import transform_to_image_format
import imageio
to_pil_image = transforms.ToPILImage()

def main():
    with torch.no_grad():
        for i in range(10):
            level = torch.load(f'./levels/generated/{i}/tensor.pt')
            new_level = edit_level(level)
            torch.save(new_level, f'./levels/edited/{i}/tensor.pt')
            image = transform_to_image_format(new_level)
            image = make_grid(image)
            image = np.array(to_pil_image(image))
            imageio.imsave(f'./levels/edited/{i}/image.png', image)


def edit_level(level):
    new_level = level
    
    #for j in range(3, 8):
    #    set_material_at_pos(new_level, 2, 3, j)

    #for k in range(3, 8):
    #    set_material_at_pos(new_level, 2, 12, k)
    
    #for l in range(3, 13):
    #    set_material_at_pos(new_level, 2, l, 3)
    
    #for m in range(3, 13):
    #    set_material_at_pos(new_level, 2, m, 7)

    for w in [0, 1, 9, 10]:
        for h in range(16):
            set_material_at_pos(new_level, 8, h, w)
    
    for h in [0, 1, 15]:
        for w in range(11):
            set_material_at_pos(new_level, 8, h, w)
    
    for w in range(11):
        if w in {4, 5, 6}:
            set_material_at_pos(new_level, 6, 14, w)
        else:
            set_material_at_pos(new_level, 8, 14, w)
    
    for w in [2,3, 7,8]:
        for h in range(16):
            set_material_at_pos(new_level, 2, h, w)
    
    return new_level

def set_material_at_pos(tensor, c, x, y):
    for i in range(10):
        if i == c:
            continue
        tensor[0, i, x, y] = 0
    
    tensor[0, c, x, y] = 1

main()