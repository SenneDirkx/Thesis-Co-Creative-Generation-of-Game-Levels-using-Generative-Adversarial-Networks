import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import models
import numpy as np
from interaction_utils.general import subsequence, make_loader, RunningSecondMoment, set_requires_grad, list_latent_to_tensor, list_level_to_tensor
import copy
from torch.utils.data import TensorDataset
import matplotlib.pyplot as plt
from interaction_utils.rewriting import estimate_v, linear_insert, og_insert
from collections import OrderedDict
from torchsummary import summary
import json

PLOT_V = False
PLOT_R = True

modelToLoad = sys.argv[1]
nz = int(sys.argv[2])
z_dims = int(sys.argv[3])
out_width = int(sys.argv[4])
out_height = int(sys.argv[5])

NB_KEYS = int(sys.argv[6])
V_ITER = int(sys.argv[7])
W_ITER = int(sys.argv[8])
C_SIZE = int(sys.argv[9])

latent_path = sys.argv[10]
edited_path = sys.argv[11]

batchSize = 1
imageSize = 32
ngf = 64
ngpu = 1
n_extra_layers = 0

# 1 2 3 of 4
CHOSEN_LAYER = int(sys.argv[12])
# 4 CONV layers (index 0, 3, 6, 9)
# Output sizes (256, 4, 4) (128, 8, 8) (64, 16, 16) (4, 32, 32)
# Names: initial:10-256:convt, pyramid:256-128:convt, pyramid:128-64:convt, final:64-4:convt
target_layer_options = [0, 3, 6, 9]
layername_options = ['initial:10-256:convt', 'pyramid:256-128:convt', 'pyramid:128-64:convt', 'final:64-4:convt']
target_layer = target_layer_options[CHOSEN_LAYER-1]
firstlayer = layername_options[CHOSEN_LAYER-1]
lastlayer = layername_options[CHOSEN_LAYER-1]

output_gen_path = sys.argv[13]
ensemble_key_i = None
try:
    ensemble_key_i = int(sys.argv[14])
    PLOT_R = False
    print("Part of an ensemble")
except:
    print("No ensemble")

print("Loading given generator...")

generator = models.DCGAN_G(imageSize, nz, z_dims, ngf, ngpu, n_extra_layers)
# This is a state dictionary that might have deprecated key labels/names
deprecatedModel = torch.load(modelToLoad, map_location=lambda storage, loc: storage)
# Make new model with weights/parameters from deprecatedModel but labels/keys from generator.state_dict()
fixedModel = OrderedDict()
for (goodKey,ignore) in generator.state_dict().items():
    # Take the good key and replace the : with . in order to get the deprecated key so the associated value can be retrieved
    badKey = goodKey.replace(":",".")
    # Some parameter settings of the generator.state_dict() are not actually part of the saved models
    if badKey in deprecatedModel:
        goodValue = deprecatedModel[badKey]
        fixedModel[goodKey] = goodValue

if not fixedModel:
    # If the fixedModel was empty, then the model was trained with the new labels, and the regular load process is fine
    generator.load_state_dict(deprecatedModel)
else:
    # Load the parameters with the fixed labels  
    generator.load_state_dict(fixedModel)

generator.eval()

summary(generator, (10,1,1))
sys.exit(0)

model = copy.deepcopy(generator)

context_model = subsequence(
            model.main, upto_layer=firstlayer,
            share_weights=True)

target_model = subsequence(
            model.main,
            first_layer=firstlayer,
            last_layer=lastlayer,
            share_weights=True)

rendering_model = subsequence(
            model.main, after_layer=lastlayer,
            share_weights=True)

first_layer = [c for c in generator.modules()
    if isinstance(c, (torch.nn.Conv2d, torch.nn.ConvTranspose2d,
                        torch.nn.Linear))][0]

seed = 1
size = C_SIZE
# 4d input if convolutional, 2d input if first layer is linear.
if isinstance(first_layer, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
    depth = first_layer.in_channels
    rng = np.random.RandomState(seed)
    zds = torch.from_numpy(
        rng.standard_normal(size * depth)
        .reshape(size, depth)).float()[:, :, None, None]
    #zds = torch.from_numpy(
    #    rng.standard_normal(size * depth * 16)
    #    .reshape(size, depth, 4, 4)).float()
else:
    depth = first_layer.in_features
    rng = np.random.RandomState(seed)
    zds = torch.from_numpy(
        rng.standard_normal(size * depth)
        .reshape(size, depth)).float()

zds = TensorDataset(zds)
print("Creating C matrix from randomly sampled k...")
with torch.no_grad():
    
    loader = make_loader(zds, None, 10)
    r2mom = RunningSecondMoment()
    for batch in loader:
        #print(batch[0].shape)
        acts = context_model(batch[0])
        #print(acts.shape)
        sample = acts.permute(0, 2, 3, 1).reshape(-1, acts.shape[1])
        r2mom.add(sample)
    
    c_matrix = r2mom.moment()

c_inverse = torch.inverse(c_matrix)

print("Loading given key examples...")
#z = torch.randn(NB_KEYS, nz, 4, 4)
z_arr = []
with open(latent_path, 'r') as latent_input:
    counter = 0

    for line in latent_input:
        if counter >= NB_KEYS:
            break
        list_level = json.loads(line)
        z_elem = list_latent_to_tensor(list_level)
        #print(z_elem.shape)
        z_arr.append(z_elem)
        counter += 1

if ensemble_key_i is not None:
    z_arr = [z_arr[ensemble_key_i]]

upscaled = False
if len(z_arr) == 1:
    upscale = 10
    print(f"PROVIDED ONLY 1 SAMPLE, UPSCALING TO {upscale}")
    z_arr = [z_arr[0] for _ in range(upscale)]
    NB_KEYS = upscale
    upscaled = True

z = torch.Tensor(NB_KEYS, z_arr[0].shape[1], z_arr[0].shape[2], z_arr[0].shape[3])
torch.cat(z_arr, out=z)

k = context_model(z).detach()
k.requires_grad = False
#print("k* shape", k.shape)
v = target_model(k).detach()
#print("v shape", v.shape)

#k_summed = k.sum(3).sum(2)
#print("k summed shape", k_summed.shape)

W0 = target_model[0].weight
print("weights", W0.shape)
# REVIEW THIS
W0_flat = W0.reshape(W0.shape[0], -1).permute(1, 0)

print("Calculating directions...")
k_arr = []
d_arr = []
d_og_arr = []
for i in range(k.shape[0]):
    k_key = k[i][None,:]
    #print(k_key.shape)
    k_arr.append(k_key)
    # REVIEW THIS
    #k_flat = k_key.permute(1, 0, 2, 3).reshape(-1, 64).permute(1, 0)
    k_flat = k_key.permute(1, 0, 2, 3).reshape(k_key.shape[1], -1)
    d = torch.matmul(c_inverse, k_flat).detach()
    d.requires_grad = False
    d_og = d.reshape(k_key.shape[1], k_key.shape[2], k_key.shape[3])[None, :]
    d_arr.append(d)
    d_og_arr.append(d_og)

#k_flat = k.permute(1, 0, 2, 3).reshape(-1, 64).permute(1, 0)
#d = torch.matmul(c_inverse, k_flat).detach()
#d.requires_grad = False
#dT = torch.transpose(d, 0, 1)
#d_og = d.reshape(64, 6, 5)[None, :]


print("Find v* of given keys...")
v_new_arr = []
x_edited_arr = []

with open(edited_path, 'r') as level_output:
    counter = 0
    for line in level_output:
        if counter >= NB_KEYS:
            break
        list_level = json.loads(line)
        #x_edited_arr.append(torch.randn(1, 10, 16, 11))
        x_edited = list_level_to_tensor(list_level)
        #print(x_edited.shape)
        x_edited_arr.append(x_edited)
        counter += 1

if ensemble_key_i is not None:
    x_edited_arr = [x_edited_arr[ensemble_key_i]]

if upscaled:
    x_edited_arr = [x_edited_arr[0] for _ in range(upscale)]

for j in range(v.shape[0]):
    print(f"-> Find v* of {j+1}th key...")
    v_key = v[i][None,:]
    v_new = estimate_v(rendering_model, v_key, x_edited_arr[i], out_width, out_height, V_ITER, plot=PLOT_V)
    v_new_arr.append(v_new)

#all_x_edited = torch.Tensor(NB_KEYS, x_edited_arr[0].shape[1], x_edited_arr[0].shape[2], x_edited_arr[0].shape[3])
#torch.cat(x_edited_arr, out=all_x_edited)

#v_new = estimate_v(rendering_model, v, all_x_edited, out_width, out_height, V_ITER, plot=True)

#assert len(k_arr) != 0 and len(v_new_arr) != 0 and len(d_og_arr) != 0

print("Merging all keys, values and directions into tensors...")

all_keys = torch.Tensor(k.shape[0], k_arr[0].shape[1], k_arr[0].shape[2], k_arr[0].shape[3])
torch.cat(k_arr, out=all_keys)

all_values = torch.Tensor(v.shape[0], v_new_arr[0].shape[1], v_new_arr[0].shape[2], v_new_arr[0].shape[3])
torch.cat(v_new_arr, out=all_values)
#all_values = v_new

all_directions = torch.Tensor(k.shape[0], d_og_arr[0].shape[1], d_og_arr[0].shape[2], d_og_arr[0].shape[3])
torch.cat(d_og_arr, out=all_directions)

print(all_keys.shape)
print(all_values.shape)
print(all_directions.shape)

print("Calculating new weights using (k*,v*) pairs...")

#weight = og_insert()
weight = linear_insert(model, target_model, all_keys, all_values, all_directions, W_ITER, plot=PLOT_R)
#print(weight.shape)

print("Rewriting model...")
print(model.main[target_layer])
model.main[target_layer].register_parameter('weight', nn.Parameter(weight))
torch.save(model.state_dict(), output_gen_path)

# print("DONE!")