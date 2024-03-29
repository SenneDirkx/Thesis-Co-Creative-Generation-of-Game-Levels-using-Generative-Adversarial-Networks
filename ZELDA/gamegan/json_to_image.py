import json
import numpy as np
import imageio
import torch
from torchvision.utils import make_grid
import torchvision.transforms as transforms
import imageio
to_pil_image = transforms.ToPILImage()
import sys

file_input = sys.argv[1]
file_output = sys.argv[2]
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
      
images = make_grid(images)
images = np.array(to_pil_image(images))
imageio.imsave(file_output, images)



# int imageHeight = list.size();
# 		int imageWidth = list.get(0).size();
# 		BufferedImage image = new BufferedImage(imageWidth, imageHeight, BufferedImage.TYPE_INT_RGB);
# 		for (int y = 0; y < imageHeight; y++) {
# 			for (int x = 0; x < imageWidth; x++) {// scans across whole image
# 				List<Integer> rgb = list.get(y).get(x);				
# 				//System.out.println(rgb);
# 				rgb.replaceAll(new UnaryOperator<Integer>() {
# 					// Clip values out of the appropriate color range
# 					@Override
# 					public Integer apply(Integer t) {
# 						// Allowable integer range for colors is [0,255]
# 						return new Integer(Math.max(0, Math.min(t, 255)));
# 					}
# 				});
# 				Color color = new Color(rgb.get(0),rgb.get(1),rgb.get(2));
# 				image.setRGB(x, y, color.getRGB());
# 			}
# 		}
# 		return image;