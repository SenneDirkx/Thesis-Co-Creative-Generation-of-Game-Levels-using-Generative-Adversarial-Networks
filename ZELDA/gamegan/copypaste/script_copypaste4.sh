#!/bin/zsh
python interact_copypaste.py ../ZeldaDungeon02_5000_10Z.pth 10 4 11 16 3 4,0,5,10 47 7,0,8,10 63,38,34,20 7,0,8,10,7,0,8,10,7,0,8,10,0,4,4,10 2500 512 latent_input.txt 4 ./output/rewritten_cp_generator4_normal.pth 16 0.0001
cd ..
echo "generating levels..."
python main.py copypaste/output/rewritten_cp_generator4_normal.pth 10 4 11 16 copypaste/latent_input.txt copypaste/output/rewritten_output4_normal.json
python json_to_image.py copypaste/output/rewritten_output4_normal.json copypaste/output/rewritten_image4_normal.png
cd ./copypaste