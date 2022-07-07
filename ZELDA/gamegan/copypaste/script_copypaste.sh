#!/bin/zsh
python interact_copypaste.py ../ZeldaDungeon02_5000_10Z.pth 10 4 11 16 6 3,3,6,6 10 6,3,10,7 11,12,18,27 6,3,10,7,6,3,10,7,6,3,10,7,6,3,10,7 2000 512 latent_input.txt 3 ./output/rewritten_cp_generator_normal.pth 8 0.0001
cd ..
echo "generating levels..."
python main.py copypaste/output/rewritten_cp_generator_normal.pth 10 4 11 16 copypaste/latent_input.txt copypaste/output/rewritten_output_normal.json
python json_to_image.py copypaste/output/rewritten_output_normal.json copypaste/output/rewritten_image_normal.png
cd ./copypaste