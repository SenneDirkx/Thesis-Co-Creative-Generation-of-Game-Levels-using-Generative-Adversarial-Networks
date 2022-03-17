import torch
import numpy as np
from torchvision.utils import make_grid
import torchvision.transforms as transforms
from output_utils import transform_to_image_format
import imageio
to_pil_image = transforms.ToPILImage()

def main():
    with torch.no_grad():
        for i in range(5):
            level = torch.load(f'./levels/generated/{i}/tensor.pt')
            new_level = edit_level(level)
            torch.save(new_level, f'./levels/edited/{i}/tensor.pt')
            image = transform_to_image_format(new_level)
            image = make_grid(image)
            image = np.array(to_pil_image(image))
            imageio.imsave(f'./levels/edited/{i}/image.png', image)


def edit_level(level):
    new_level = level
    
    for j in range(3, 8):
        set_material_at_pos(new_level, 2, 3, j)

    for k in range(3, 8):
        set_material_at_pos(new_level, 2, 12, k)
    
    for l in range(3, 13):
        set_material_at_pos(new_level, 2, l, 3)
    
    for m in range(3, 13):
        set_material_at_pos(new_level, 2, m, 7)
    
    return new_level

def set_material_at_pos(tensor, c, x, y):
    for i in range(10):
        if i == c:
            continue
        tensor[0, i, x, y] = 0
    
    tensor[0, c, x, y] = 1

main()