import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from model_generator import GeneratorConv, gumbel_softmax
import numpy as np
from utils import subsequence, make_loader, RunningSecondMoment, set_requires_grad
import copy
from torch.utils.data import TensorDataset
import matplotlib.pyplot as plt
from interaction_utils import estimate_v, linear_insert, og_insert

KEY_ID = 0
NB_KEYS = 5
V_ITER = 10000
W_ITER = 10000

nz = 128
temperature = 1.0

print("Loading given generator...")
generator = GeneratorConv(nz, temperature)
generator.load_state_dict(torch.load("./generator.pth"))
generator.eval()

target_layer = 3
firstlayer = f'{target_layer}'
lastlayer = f'{target_layer}'

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
size = 100
# 4d input if convolutional, 2d input if first layer is linear.
if isinstance(first_layer, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
    depth = first_layer.in_channels
    rng = np.random.RandomState(seed)
    zds = torch.from_numpy(
        rng.standard_normal(size * depth * 16)
        .reshape(size, depth, 4, 4)).float()
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
        acts = context_model(batch[0])
        sample = acts.permute(0, 2, 3, 1).reshape(-1, acts.shape[1])
        r2mom.add(sample)
    
    c_matrix = r2mom.moment()

c_inverse = torch.inverse(c_matrix)

print("Loading given key examples...")
#z = torch.randn(NB_KEYS, nz, 4, 4)
z_arr = []
for i in range(NB_KEYS):
    z_elem = torch.load(f'./levels/generated/{i}/latent.pt')
    z_arr.append(z_elem)

z = torch.Tensor(NB_KEYS, z_arr[0].shape[1], z_arr[0].shape[2], z_arr[0].shape[3])
torch.cat(z_arr, out=z)

k = context_model(z).detach()
k.requires_grad = False
#print("k* shape", k.shape)
v = target_model(k).detach()
#print("v shape", v.shape)
v_flat = v.view(-1)
x = gumbel_softmax(rendering_model(v), temperature).detach()

#k_summed = k.sum(3).sum(2)
#print("k summed shape", k_summed.shape)

W0 = target_model[0].weight
print("weights", W0.shape)
W0_flat = W0.reshape(64, -1).permute(1, 0)

print("Calculating directions...")
k_arr = []
d_arr = []
d_og_arr = []
for i in range(NB_KEYS):
    k_key = k[i][None,:]
    k_arr.append(k_key)
    k_flat = k_key.permute(1, 0, 2, 3).reshape(-1, 64).permute(1, 0)
    d = torch.matmul(c_inverse, k_flat).detach()
    d.requires_grad = False
    d_og = d.reshape(64, 6, 5)[None, :]
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

for i in range(NB_KEYS):
    #x_edited_arr.append(torch.randn(1, 10, 16, 11))
    x_edited = torch.load(f'./levels/edited/{i}/tensor.pt')
    x_edited_arr.append(x_edited)

for j in range(NB_KEYS):
    print(f"-> Find v* of {j+1}th key...")
    v_key = v[i][None,:]
    v_new = estimate_v(rendering_model, v_key, x_edited_arr[i], temperature, V_ITER)
    v_new_arr.append(v_new)

#print('v_sum', v_sum)
#print('x_diff_begin', x_diff_begin)
#print('x_diff_opt', x_diff_opt)

#plt.figure()
#plt.title("V losses")
#plt.plot(v_losses)
#plt.show()
assert len(k_arr) != 0 and len(v_new_arr) != 0 and len(d_og_arr) != 0

print("Merging all keys, values and directions into tensors...")

all_keys = torch.Tensor(NB_KEYS, k_arr[0].shape[1], k_arr[0].shape[2], k_arr[0].shape[3])
torch.cat(k_arr, out=all_keys)

all_values = torch.Tensor(NB_KEYS, v_new_arr[0].shape[1], v_new_arr[0].shape[2], v_new_arr[0].shape[3])
torch.cat(v_new_arr, out=all_values)

all_directions = torch.Tensor(NB_KEYS, d_og_arr[0].shape[1], d_og_arr[0].shape[2], d_og_arr[0].shape[3])
torch.cat(d_og_arr, out=all_directions)

print(all_keys.shape)
print(all_values.shape)
print(all_directions.shape)

print("Calculating new weights using (k*,v*) pairs...")

#weight = og_insert()
weight = linear_insert(model, target_model, all_keys, all_values, all_directions, W_ITER)
#print(weight.shape)

print("Rewriting model...")
model.main[target_layer].register_parameter('weight', nn.Parameter(weight))
torch.save(model.state_dict(), './rewritten_generator.pth')

print("DONE!")