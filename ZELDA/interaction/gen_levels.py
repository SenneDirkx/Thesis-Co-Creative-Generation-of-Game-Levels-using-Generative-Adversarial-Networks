import torch
from model_generator import GeneratorConv, GeneratorConv2
from torchvision.utils import make_grid
import numpy as np
import torchvision.transforms as transforms
from output_utils import transform_to_image_format
import imageio

import sys

gen_filename = sys.argv[1]
print(gen_filename)
to_pil_image = transforms.ToPILImage()

nz = 128
temperature = 1.0

generator = GeneratorConv(nz, temperature)
generator.load_state_dict(torch.load(gen_filename))
generator.eval()

for i in range(10):
    rnd_input = torch.randn(1, nz, 4, 4)
    result = generator(rnd_input, temperature, False)
    image = transform_to_image_format(result)
    image = make_grid(image)
    image = np.array(to_pil_image(image))
    torch.save(result, f'./levels/generated/{i}/tensor.pt')
    torch.save(rnd_input, f'./levels/generated/{i}/latent.pt')
    imageio.imsave(f'./levels/generated/{i}/image.png', image)