import json
import numpy as np
import imageio
import torch
from torchvision.utils import make_grid
import torchvision.transforms as transforms
import imageio
to_pil_image = transforms.ToPILImage()

counter = 0
num_lines = sum(1 for _ in open('level_output.json'))
images = torch.zeros((num_lines, 3, 16, 11), dtype=torch.uint8)
with open('level_output.json', 'r') as level_output:
    
    for line in level_output:
        list_level = json.loads(line)
        height = len(list_level)
        width = len(list_level[0])
        image = torch.zeros((height, width, 3), dtype=torch.uint8)
        for y in range(height):
            for x in range(width):
                tile = list_level[y][x]
                if tile == 0:
                    image[y,x] = torch.tensor([175,175,175], dtype=torch.uint8)
                elif tile == 1:
                    image[y,x] = torch.tensor((54, 29, 102), dtype=torch.uint8)
                elif tile == 2:
                    image[y,x] = torch.tensor((255, 0, 51), dtype=torch.uint8)
                elif tile == 3:
                    image[y,x] = torch.tensor((87, 25, 15), dtype=torch.uint8)
                else:
                    image[y,x] = torch.tensor((255,255,255), dtype=torch.uint8)
        images[counter] = image.permute(2,0,1)
        counter += 1
      
images = make_grid(images)
images = np.array(to_pil_image(images))
imageio.imsave('test.png', images)



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