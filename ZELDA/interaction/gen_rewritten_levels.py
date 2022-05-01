import torch
from model_generator import GeneratorConv, GeneratorConv2
from torchvision.utils import make_grid
import numpy as np
import torchvision.transforms as transforms
from output_utils import transform_to_image_format
import imageio
to_pil_image = transforms.ToPILImage()

nz = 128
temperature = 1.0

generator = GeneratorConv2(nz, temperature)
generator.load_state_dict(torch.load("./rewritten_generator.pth"))
generator.eval()

for i in range(5):
    rnd_input = torch.randn(1, nz)
    result, _ = generator(rnd_input, temperature, False)
    image = transform_to_image_format(result)
    image = make_grid(image)
    image = np.array(to_pil_image(image))
    torch.save(result, f'./levels/rewritten/{i}/tensor.pt')
    imageio.imsave(f'./levels/rewritten/{i}/image.png', image)