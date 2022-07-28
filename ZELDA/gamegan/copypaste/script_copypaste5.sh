#!/bin/zsh
python interact_copypaste.py ../ZeldaDungeon02_5000_10Z.pth 10 4 11 16 62 5,4,6,5 18 6,3,10,7 10,11,12,54 6,3,10,7,6,3,10,7,6,3,10,7,6,3,10,7 2500 1024 latent_input.txt 3 ./output/rewritten_cp_generator5_normal.pth 1 0.001
cd ..
echo "generating levels..."
python main.py copypaste/output/rewritten_cp_generator5_normal.pth 10 4 11 16 copypaste/latent_input.txt copypaste/output/rewritten_output5_normal.json
python json_to_image.py copypaste/output/rewritten_output5_normal.json copypaste/output/rewritten_image5_normal.png
cd ./copypaste