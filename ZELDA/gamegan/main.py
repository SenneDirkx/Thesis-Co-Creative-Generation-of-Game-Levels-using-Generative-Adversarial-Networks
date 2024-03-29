# This generator program expands a low-dimentional latent vector into a 2D array of tiles.
# Each line of input should be an array of z vectors (which are themselves arrays of floats -1 to 1)
# Each line of output is an array of 32 levels (which are arrays-of-arrays of integer tile ids)

import torch
from torch.autograd import Variable

import sys
import json
import numpy
import models
from collections import OrderedDict

if __name__ == '__main__':
    
    modelToLoad = sys.argv[1]
    nz = int(sys.argv[2])
    z_dims = int(sys.argv[3])
    out_width = int(sys.argv[4])
    out_height = int(sys.argv[5])

    load_path = sys.argv[6]
    save_path = sys.argv[7]

#    if z_dims == 4 : # Assume this is Zelda (4 tiles, currently)
#        out_height = 16
#        out_width = 11
#    elif z_dims == 6 or z_dims == 3: # Fixed Zelda (rotated)
#        # The fixed Zelda rotates the rooms to match the original game presentation
#        out_height = 11
#        out_width = 16
#    elif z_dims == 8: #Lode Runner
#        out_height = 22
#        out_width = 32 
#    else: # Assume this is Mario (10 or 13 tiles, depending)
#       out_height = 14
#       out_width = 28

    batchSize = 1
    #nz = 10 #Dimensionality of latent vector

    imageSize = 32
    ngf = 64
    ngpu = 1
    n_extra_layers = 0

   
    generator = models.DCGAN_G(imageSize, nz, z_dims, ngf, ngpu, n_extra_layers)
    #print(generator.state_dict()) 
    # This is a state dictionary that might have deprecated key labels/names
    deprecatedModel = torch.load(modelToLoad, map_location=lambda storage, loc: storage)
    #print(deprecatedModel)
    # Make new model with weights/parameters from deprecatedModel but labels/keys from generator.state_dict()
    fixedModel = OrderedDict()
    for (goodKey,ignore) in generator.state_dict().items():
        # Take the good key and replace the : with . in order to get the deprecated key so the associated value can be retrieved
        badKey = goodKey.replace(":",".")
        #print(goodKey)
        #print(badKey)
        # Some parameter settings of the generator.state_dict() are not actually part of the saved models
        if badKey in deprecatedModel:
            goodValue = deprecatedModel[badKey]
            fixedModel[goodKey] = goodValue

    if not fixedModel:
        #print("LOAD REGULAR")
        #print(deprecatedModel)
        # If the fixedModel was empty, then the model was trained with the new labels, and the regular load process is fine
        generator.load_state_dict(deprecatedModel)
    else:
        # Load the parameters with the fixed labels  
        generator.load_state_dict(fixedModel)


    generated_levels = []

    with torch.no_grad():
        with open(load_path, 'r') as latent_input:

            for line in latent_input:
                # "0\n" secret exit command
                # This breaks the conditional GAN when class 0 is used
                #if len(line)==2 and int(line)==0:
                #    break

                
                # Standard GAN. Input is just latent vector
                lv = numpy.array(json.loads(line))
                latent_vector = torch.FloatTensor( lv ).view(batchSize, nz, 1, 1) 
                levels = generator(Variable(latent_vector))

                #levels.data = levels.data[:,:,:14,:28] #Cut of rest to fit the 14x28 tile dimensions
                levels.data = levels.data[:, :, :out_height, :out_width]
                #vutils.save_image(levels.data, "test.png")
                levels  = levels.data.cpu().numpy()
                #Cut of rest to fit the 14x28 tile dimensions
                levels = numpy.argmax(levels, axis = 1)
                
        
                #levels.data[levels.data > 0.] = 1  #SOLID BLOCK
                #levels.data[levels.data < 0.] = 2  #EMPTY TILE

                # Jacob: Only output first level, since we are only really evaluating one at a time
                generated_levels.append(json.dumps(levels[0].tolist()))

                # [[[[1]], [[1]], [[1]], [[1]], [[1]], [[1]], [[1]], [[1]], [[1]], [[1]]]]

                # [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 3, 3, 3, 1, 1, 1, 1], [1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1], [1, 1, 0, 2, 0, 2, 0, 2, 0, 1, 1], [1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1], [1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1], [1, 1, 0, 2, 0, 2, 0, 2, 0, 1, 1], [1, 3, 0, 0, 0, 0, 0, 0, 0, 3, 1], [1, 3, 0, 0, 0, 0, 0, 0, 0, 3, 1], [1, 1, 0, 2, 0, 2, 0, 2, 0, 1, 1], [1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1], [1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1], [1, 1, 0, 2, 0, 2, 0, 2, 0, 1, 1], [1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1], [1, 1, 1, 1, 3, 3, 3, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]

        with open(save_path, 'w') as level_output:
            level_output.writelines(map(lambda x: x + "\n", generated_levels))