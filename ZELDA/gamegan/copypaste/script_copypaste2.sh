#!/bin/zsh
python interact_copypaste.py ../ZeldaDungeon02_5000_10Z.pth 10 4 11 16 58 2,2,5,8 6 10,2,13,8 47,38,6,63 10,2,13,8,10,2,13,8,10,2,13,8,10,2,13,8 5000 1024 latent_input.txt 3 ./output/rewritten_cp_generator2_normal.pth 1 0.0001
cd ..
echo "generating levels..."
python main.py copypaste/output/rewritten_cp_generator2_normal.pth 10 4 11 16 copypaste/latent_input.txt copypaste/output/rewritten_output2_normal.json
python json_to_image.py copypaste/output/rewritten_output2_normal.json copypaste/output/rewritten_image2_normal.png
cd ./copypaste