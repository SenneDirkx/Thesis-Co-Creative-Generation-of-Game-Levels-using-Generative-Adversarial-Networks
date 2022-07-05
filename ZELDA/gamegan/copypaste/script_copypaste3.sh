#!/bin/zsh
python interact_copypaste.py ../ZeldaDungeon02_5000_10Z.pth 10 4 11 16 53 2,2,6,8 4 9,2,13,8 7,16,17,26,55 9,2,13,8,9,2,13,8,9,2,13,8,9,2,13,8,9,2,13,8 1000 512 latent_input.txt 3 ./rewritten_cp_generator3.pth 128 0.001
cd ..
echo "generating levels..."
python main.py copypaste/rewritten_cp_generator3.pth 10 4 11 16 copypaste/latent_input.txt copypaste/rewritten_output3.json
python json_to_image.py copypaste/rewritten_output3.json copypaste/rewritten_image3.png
cd ./copypaste