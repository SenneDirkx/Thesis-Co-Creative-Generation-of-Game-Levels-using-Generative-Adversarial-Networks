import torch
from model_generator import GeneratorConv
from torchvision.utils import make_grid
import numpy as np
import torchvision.transforms as transforms
from output_utils import transform_to_image_format
import imageio
to_pil_image = transforms.ToPILImage()

nz = 128
temperature = 1.0

generator = GeneratorConv(nz, temperature)
generator.load_state_dict(torch.load("./generator.pth"))
generator.eval()

for i in range(5):
    rnd_input = torch.randn(1, nz, 4, 4)
    result = generator(rnd_input, temperature, False)
    image = transform_to_image_format(result)
    image = make_grid(image)
    image = np.array(to_pil_image(image))
    torch.save(result, f'./levels/generated/{i}/tensor.pt')
    imageio.imsave(f'./levels/generated/{i}/image.png', image)