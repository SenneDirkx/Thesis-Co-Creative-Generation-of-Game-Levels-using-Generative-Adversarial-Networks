#!/bin/zsh
python interact.py ./ZeldaDungeon02_5000_10Z.pth 10 4 11 16 5 2,2,9,5 7,40,46,30 2,2,9,5,2,2,9,5,2,2,9,5,2,2,9,5 10000 10000 1024 latent_input.txt edited_output4.json 3 ./output_custom_local/rewritten_custom_generator4.pth 1 0.0001
echo "generating levels..."
python main.py ./output_custom_local/rewritten_custom_generator4.pth 10 4 11 16 ./latent_input.txt ./output_custom_local/rewritten_output4.json
python json_to_image.py ./output_custom_local/rewritten_output4.json ./output_custom_local/rewritten_image4.png