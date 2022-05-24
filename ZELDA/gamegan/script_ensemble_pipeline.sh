#!/bin/zsh
pipenv shell

echo "generating latent input"
python generate_random_latent.py latent_input.txt 64

echo "creating og levels"
python main.py ./ZeldaDungeon02_5000_10Z.pth 10 4 11 16 latent_input.txt level_output.json
python json_to_image.py level_output.json og_image.png

echo "editing levels"
python generate_random_latent.py interact_input.txt 10
python main.py ./ZeldaDungeon02_5000_10Z.pth 10 4 11 16 interact_input.txt interact_level_output.json
python editor.py interact_level_output.json edited_level_output.json
python json_to_image.py edited_level_output.json edited_image.png

echo "performing interaction"
for i in {0..9}
do
    echo "Interaction $i"
    python interact.py ./ZeldaDungeon02_5000_10Z.pth 10 4 11 16 10 50 50 512 interact_input.txt edited_level_output.json 3 ./ensemble/rewritten_gen_$i.pth $i
done

echo "creating rewritten levels"
python ensemble_main.py 10 10 4 11 16 latent_input.txt ensemble_rewritten_level_output.json
python json_to_image.py ensemble_rewritten_level_output.json ensemble_rewritten_image.png