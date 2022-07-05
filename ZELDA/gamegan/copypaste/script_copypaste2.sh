#!/bin/zsh
python interact_copypaste.py ../ZeldaDungeon02_5000_10Z.pth 10 4 11 16 58 2,2,5,8 6 10,2,13,8 46,55,6,63 10,2,13,8,10,2,13,8,10,2,13,8,10,2,13,8 1000 512 latent_input.txt 3 ./rewritten_cp_generator2.pth 128 0.001
cd ..
echo "generating levels..."
python main.py copypaste/rewritten_cp_generator2.pth 10 4 11 16 copypaste/latent_input.txt copypaste/rewritten_output2.json
python json_to_image.py copypaste/rewritten_output2.json copypaste/rewritten_image2.png
cd ./copypaste