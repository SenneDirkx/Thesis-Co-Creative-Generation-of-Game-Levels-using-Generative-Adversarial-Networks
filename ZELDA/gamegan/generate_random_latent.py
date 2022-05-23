import numpy as np
import json
import sys

filename = sys.argv[1]
NB_LEVELS = int(sys.argv[2])

with open(filename, 'w') as latent_input:
    for i in range(NB_LEVELS):
        latent = np.random.rand(1,10,1,1) * 2 - 1
        latent_list = latent.tolist()
        latent_str = json.dumps(latent_list)
        latent_input.write(latent_str + "\n")