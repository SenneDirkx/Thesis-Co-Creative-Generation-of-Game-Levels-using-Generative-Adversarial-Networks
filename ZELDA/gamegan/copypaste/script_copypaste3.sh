#!/bin/zsh
python interact_copypaste.py ../ZeldaDungeon02_5000_10Z.pth 10 4 11 16 53 2,2,6,8 4 9,2,13,8 7,16,17,26,55 9,2,13,8,9,2,13,8,9,2,13,8,9,2,13,8,9,2,13,8 5000 1024 latent_input.txt 3 ./output/rewritten_cp_generator3_normal.pth 1 0.0001
cd ..
echo "generating levels..."
python main.py copypaste/output/rewritten_cp_generator3_normal.pth 10 4 11 16 copypaste/latent_input.txt copypaste/output/rewritten_output3_normal.json
python json_to_image.py copypaste/output/rewritten_output3_normal.json copypaste/output/rewritten_image3_normal.png
cd ./copypaste