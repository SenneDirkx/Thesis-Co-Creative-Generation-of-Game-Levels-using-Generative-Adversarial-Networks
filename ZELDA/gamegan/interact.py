import warnings
warnings.simplefilter("ignore", UserWarning)
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import models
import numpy as np
from interaction_utils.general import load_edited_x, subsequence, make_loader, RunningSecondMoment, set_requires_grad, list_latent_to_tensor, list_level_to_tensor, zca_from_cov, zca_whitened_query_key
import copy
from torch.utils.data import TensorDataset
import matplotlib.pyplot as plt
from interaction_utils.rewriting import estimate_v, linear_insert, og_insert, normal_insert_fixed
from collections import OrderedDict
from torchsummary import summary
import json

PLOT_V = True
PLOT_R = False
OUTPUT_DIM_RAW = 32

modelToLoad = sys.argv[1]
nz = int(sys.argv[2])
z_dims = int(sys.argv[3])
out_width = int(sys.argv[4])
out_height = int(sys.argv[5])

PASTE_KEY = int(sys.argv[6])
t_p, l_p, b_p, r_p = [int(bnd) for bnd in sys.argv[7].split(",")]
CONTEXT_KEYS = [int(ck) for ck in sys.argv[8].split(",")]
context_bounds = [int(bnd) for bnd in sys.argv[9].split(",")]

V_ITER = int(sys.argv[10])

W_ITER = int(sys.argv[11])
C_SIZE = int(sys.argv[12])

latent_path = sys.argv[13]
edited_path = sys.argv[14]

batchSize = 1
imageSize = 32
ngf = 64
ngpu = 1
n_extra_layers = 0

# 1 2 3 of 4
CHOSEN_LAYER = int(sys.argv[15])
# 4 CONV layers (index 0, 3, 6, 9)
# Output sizes (256, 4, 4) (128, 8, 8) (64, 16, 16) (4, 32, 32)
# Names: initial:10-256:convt, pyramid:256-128:convt, pyramid:128-64:convt, final:64-4:convt
target_layer_options = [0, 3, 6, 9]
layername_options = ['initial:10-256:convt', 'pyramid:256-128:convt', 'pyramid:128-64:convt', 'final:64-4:convt']
target_layer = target_layer_options[CHOSEN_LAYER-1]
firstlayer = layername_options[CHOSEN_LAYER-1]
lastlayer = layername_options[CHOSEN_LAYER-1]

output_gen_path = sys.argv[16]
DRANK = int(sys.argv[17])
LR = float(sys.argv[18])

# full, inplace, moving, half
COPY_MODE = 'moving'
# svd, zca
RANK_METHOD = 'zca'

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

#summary(generator, (10,1,1))
#sys.exit(0)

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
zca_matrix = zca_from_cov(c_matrix)

#print("Loading given key examples...")
#z = torch.randn(NB_KEYS, nz, 4, 4)
paste_z_arr = []
context_z_arr = []
with open(latent_path, 'r') as latent_input:
    counter = 0

    for line in latent_input:
        list_level = json.loads(line)
        z_elem = list_latent_to_tensor(list_level)
        #print(z_elem.shape)
        if counter == PASTE_KEY:
            paste_z_arr.append(z_elem)
        if counter in CONTEXT_KEYS:
            context_z_arr.append(z_elem)
        counter += 1

paste_z = torch.Tensor(1, paste_z_arr[0].shape[1], paste_z_arr[0].shape[2], paste_z_arr[0].shape[3])
torch.cat(paste_z_arr, out=paste_z)

context_z = torch.Tensor(1, context_z_arr[0].shape[1], context_z_arr[0].shape[2], context_z_arr[0].shape[3])
torch.cat(context_z_arr, out=context_z)

paste_k = context_model(paste_z).detach()
paste_k.requires_grad = False
paste_v = target_model(paste_k).detach()

context_k = context_model(context_z).detach()
context_k.requires_grad = False

W0 = target_model[0].weight
#print("weights", W0.shape)
# REVIEW THIS
W0_flat = W0.reshape(W0.shape[0], -1).permute(1, 0)

#print("Calculating directions...")
import copypaste.cp_utils as cp_utils

d_arr = []
d_og_arr = []
zca_arr = []
for i in range(context_k.shape[0]):
    k_key = context_k[i][None,:]
    k_context_mask = torch.zeros((1, 1, OUTPUT_DIM_RAW, OUTPUT_DIM_RAW), dtype=torch.float32)

    for mi in range(out_height):
        for mj in range(out_width):
            if mi >= context_bounds[i*4] and mi <= context_bounds[i*4+2] and mj >= context_bounds[i*4+1] and mj <= context_bounds[i*4+3]:
                k_context_mask[0, 0, mi, mj] = 1
    
    #print(k_context_mask[0,0])
    interpolated_k_context_mask = cp_utils.extract_interpolated_mask(k_context_mask, k_key.shape[2:])

    k_obs = k_key.permute(0, 2, 3, 1).reshape(-1, k_key.shape[1])
    k_w = interpolated_k_context_mask[0,0].view(-1)[:, None]
    d_zca_full = (k_w * zca_whitened_query_key(zca_matrix, k_obs))
    d_zca = d_zca_full[(k_w > 0).nonzero()[:, 0], :]
    zca_arr.append(d_zca)

    #print(interpolated_k_context_mask[0,0])
    masked_k_key = torch.mul(k_key, interpolated_k_context_mask)
    #print(masked_k_key[0,0])

    k_flat = masked_k_key.permute(1, 0, 2, 3).reshape(masked_k_key.shape[1], -1)
    d = torch.matmul(c_inverse, k_flat).detach()
    d.requires_grad = False
    d_og = d.reshape(masked_k_key.shape[1], masked_k_key.shape[2], masked_k_key.shape[3])[None, :]
    d = d.permute(1, 0)
    #print(d_og[0,0])
    d_arr.append(d)
    d_og_arr.append(d_og)



#print("Find v* of given keys...")

paste_mask = torch.zeros((1, 1, OUTPUT_DIM_RAW, OUTPUT_DIM_RAW), dtype=torch.float32)


for mi in range(out_height):
    for mj in range(out_width):
        if mi >= t_p and mi <= b_p and mj >= l_p and mj <= r_p:
            paste_mask[0, 0, mi, mj] = 1
                
interpolated_k_paste_mask = cp_utils.extract_interpolated_mask(paste_mask, paste_k.shape[2:])
interpolated_v_custom_mask = cp_utils.extract_interpolated_mask(paste_mask, paste_v.shape[2:])
tv, lv, bv, rv = cp_utils.positive_bounding_box(interpolated_v_custom_mask[0,0])
print("V bounding box:", tv, lv, bv, rv)
clip_mask = interpolated_v_custom_mask[:, :, tv:bv, lv:rv]


random_v = torch.rand(paste_v.shape)
edited_x = load_edited_x(edited_path)
v_new = estimate_v(rendering_model, random_v, edited_x, out_width, out_height, V_ITER, plot=PLOT_V)

clip = v_new[:, :, tv:bv, lv:rv]
center = (tv + bv) // 2, (lv + rv) // 2

ttv, ltv = (max(0, min(e - s, c - s // 2))
            for s, c, e in zip(clip.shape[2:], center, paste_v.shape[2:]))
btv, rtv = ttv + clip.shape[2], ltv + clip.shape[3]


source_k = paste_k
target_v = paste_v.clone()
target_v[:, :, ttv:btv, ltv:rtv] = (1 - clip_mask) * target_v[:, :, ttv:btv, ltv:rtv] + clip_mask * clip

vr, hr = [ts // ss for ts, ss in zip(target_v.shape[2:], source_k.shape[2:])]
st, sl, sb, sr = ttv // vr, ltv // hr, -(-btv // vr), -(-rtv // hr)
tt, tl, tb, tr = st * vr, sl * hr, sb * vr, sr * hr
print("Goal in bounding box:", st, sl, sb, sr)
print("Goal out bounding box:", tt, tl, tb, tr)
cs, ct = source_k[:, :, st:sb, sl:sr], target_v[:, :, tt:tb, tl:tr]

goal_in = cs
goal_out = ct



all_context = torch.cat([di for di in d_arr])
all_zca = torch.cat(zca_arr)

if RANK_METHOD == 'svd':
    just_avg = all_context.mean(0)
    u, s, v = torch.svd(all_context.permute(1, 0), some=False)
    if (just_avg * u[:, 0]).sum() < 0:
        # Flip the first singular vectors to agree with avg direction
        u[:, 0] = -u[:, 0]
        v[:, 0] = -v[:, 0]
    if u.shape[1] < DRANK:
        print("No SVD applied")
        final_context = all_context
    else:
        print("SVD rank reducing to", DRANK)
        final_context = u.permute(1, 0)[:DRANK]
elif RANK_METHOD == 'zca':
    _, _, q = all_zca.svd(compute_uv=True)
    top_e_vec = q[:, :DRANK]
    row_dirs = zca_whitened_query_key(zca_matrix, top_e_vec.t())
    just_avg = (all_zca).sum(0)
    q, r = torch.qr(row_dirs.permute(1, 0))
    signs = (q * just_avg[:, None]).sum(0).sign()
    q = q * signs[None, :]
    final_context = q.permute(1, 0)
else:
    print("no rank reduction")
    final_context = all_context

#print("final_context", final_context.shape)
#sys.exit(0)


print("Calculating new weights using (k*,v*) pairs...")

print("Goal in:", goal_in.shape)
print("Goal out:", goal_out.shape)
print("Context:", final_context.shape)

#weight = og_insert()
loss_begin, loss_end = normal_insert_fixed(target_model, goal_in, goal_out, final_context, W_ITER=W_ITER, P_ITER=4, lr=LR)
#weight = linear_insert(model, target_model, all_keys, all_values, all_directions, W_ITER, plot=PLOT_R)
#print(weight.shape)

#print("Rewriting model...")
model.main[target_layer].register_parameter('weight', nn.Parameter(target_model[0].weight))
torch.save(model.state_dict(), output_gen_path)

# print("DONE!")