#!/bin/zsh
python interact.py ./ZeldaDungeon02_5000_10Z.pth 10 4 11 16 28 5,2,11,8 17,63,59,21 5,2,11,8,5,2,11,8,5,2,11,8,5,2,11,8 10000 10000 1024 latent_input.txt edited_output.json 3 ./output_custom_local/rewritten_custom_generator.pth 1 0.0001
echo "generating levels..."
python main.py ./output_custom_local/rewritten_custom_generator.pth 10 4 11 16 ./latent_input.txt ./output_custom_local/rewritten_output.json
python json_to_image.py ./output_custom_local/rewritten_output.json ./output_custom_local/rewritten_image.png