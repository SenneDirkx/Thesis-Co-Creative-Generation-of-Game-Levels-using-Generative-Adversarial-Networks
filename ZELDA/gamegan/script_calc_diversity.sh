#!/bin/zsh
pipenv shell
echo "Three estimations incoming"
python generate_random_latent.py 100
python main.py ./ZeldaDungeon02_5000_10Z.pth 10 4 11 16
python diversity.py 1000
python generate_random_latent.py 100
python main.py ./ZeldaDungeon02_5000_10Z.pth 10 4 11 16
python diversity.py 1000
python generate_random_latent.py 100
python main.py ./ZeldaDungeon02_5000_10Z.pth 10 4 11 16
python diversity.py 1000