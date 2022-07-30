#!/bin/zsh
python interact.py ./ZeldaDungeon02_5000_10Z.pth 10 4 11 16 43 3,3,7,8 42,44 3,3,7,8,3,3,7,8 10000 10000 1024 latent_input.txt edited_output2.json 3 ./output_custom_local/rewritten_custom_generator2.pth 1 0.0001
echo "generating levels..."
python main.py ./output_custom_local/rewritten_custom_generator2.pth 10 4 11 16 ./latent_input.txt ./output_custom_local/rewritten_output2.json
python json_to_image.py ./output_custom_local/rewritten_output2.json ./output_custom_local/rewritten_image2.png