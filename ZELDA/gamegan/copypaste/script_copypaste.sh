#!/bin/zsh
python interact_copypaste.py ../ZeldaDungeon02_5000_10Z.pth 10 4 11 16 6 3,3,6,6 10 6,3,10,7 11,12,18,27 6,3,10,7,6,3,10,7,6,3,10,7,6,3,10,7 1000 512 latent_input.txt 3 ./rewritten_cp_generator.pth 250 0.0001
cd ..
echo "generating levels..."
python main.py copypaste/rewritten_cp_generator.pth 10 4 11 16 copypaste/latent_input.txt copypaste/rewritten_output.json
python json_to_image.py copypaste/rewritten_output.json copypaste/rewritten_image.png
cd ./copypaste