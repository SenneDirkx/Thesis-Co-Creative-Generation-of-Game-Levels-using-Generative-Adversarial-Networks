#!/bin/zsh
python interact.py ./ZeldaDungeon02_5000_10Z.pth 10 4 11 16 56 3,3,6,7 51,16 3,3,6,7,3,3,6,7 10000 10000 1024 latent_input.txt edited_output3.json 3 ./output_custom_local/rewritten_custom_generator3.pth 1 0.0001
echo "generating levels..."
python main.py ./output_custom_local/rewritten_custom_generator3.pth 10 4 11 16 ./latent_input.txt ./output_custom_local/rewritten_output3.json
python json_to_image.py ./output_custom_local/rewritten_output3.json ./output_custom_local/rewritten_image3.png