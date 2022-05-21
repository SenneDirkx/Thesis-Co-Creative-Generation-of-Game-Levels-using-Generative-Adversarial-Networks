# This generator program expands a low-dimentional latent vector into a 2D array of tiles.
# Each line of input should be an array of z vectors (which are themselves arrays of floats -1 to 1)
# Each line of output is an array of 32 levels (which are arrays-of-arrays of integer tile ids)

import torch
import torchvision.utils as vutils
from torch.autograd import Variable

import sys
import json
import numpy
import models
import matplotlib.pyplot as plt
import math

import random
from collections import OrderedDict

def combine_images(generated_images):
    num = generated_images.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num)/width))
    shape = generated_images.shape[1:]
    image = numpy.zeros((height*shape[0], width*shape[1],shape[2]), dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index/width)
        j = index % width
        image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1]] = img
    return image

if __name__ == '__main__':
    
    modelToLoad = sys.argv[1]
    nz = int(sys.argv[2])
    z_dims = int(sys.argv[3])
    out_width = int(sys.argv[4])
    out_height = int(sys.argv[5])
    if len(sys.argv)<=6:
        num_classes = -1
    else: 
        num_classes = int(sys.argv[6])

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

    testing = False

    if testing:  
        line = []
        for i in range (batchSize):
            line.append( [ random.uniform(-1.0, 1.0) ]*nz )

        #This is the format that we expect from sys.stdin
        print(line)
        line = json.dumps(line)
        lv = numpy.array(json.loads(line))
        latent_vector = torch.FloatTensor( lv ).view(batchSize, nz, 1, 1) #torch.from_numpy(lv)# torch.FloatTensor( torch.from_numpy(lv) )
        #latent_vector = numpy.array(json.loads(line))
        levels = generator(Variable(latent_vector, volatile=True))
        im = levels.data.cpu().numpy()
        im = im[:,:,:out_height,:out_width] #Cut off rest to fit the 14x28 tile dimensions
        im = numpy.argmax( im, axis = 1)
        #print(json.dumps(levels.data.tolist()))
        print("Saving to file ")
        im = ( plt.get_cmap('rainbow')( im/float(z_dims) ) )

        plt.imsave('fake_sample.png', combine_images(im) )

        exit()

    generated_levels = []

    with torch.no_grad():
        with open('latent_input.txt', 'r') as latent_input:

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

        with open('level_output.json', 'w') as level_output:
            level_output.writelines(map(lambda x: x + "\n", generated_levels))