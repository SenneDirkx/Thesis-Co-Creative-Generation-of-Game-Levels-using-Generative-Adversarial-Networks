import shutil
import sys

run = int(sys.argv[1])

images = ['og_image.png', 'edited_image.png', 'rewritten_image.png']
latents = ['latent_input.txt', 'interact_input.txt']
levels = ['level_output.json', 'edited_level_output.json', 'interact_level_output.json', 'rewritten_level_output.json']
generator = ['rewritten_generator.pth']

files = images + latents + levels + generator

for f in files:
    shutil.copy(f, f'interaction_data/run{run}/')