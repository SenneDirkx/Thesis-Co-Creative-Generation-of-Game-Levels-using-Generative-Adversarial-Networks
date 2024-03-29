import warnings
warnings.simplefilter("ignore", UserWarning)
import sys
sys.path.append("..")
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as VF
import models
import numpy as np
from interaction_utils.general import subsequence, make_loader, RunningSecondMoment, set_requires_grad, list_latent_to_tensor, list_level_to_tensor, zca_from_cov, zca_whitened_query_key
import copy
from torch.utils.data import TensorDataset
import matplotlib.pyplot as plt
from interaction_utils.rewriting import estimate_v, linear_insert, og_insert, linear_insert_fixed, normal_insert_fixed
from collections import OrderedDict
from torchsummary import summary
import json

PLOT_V = False
PLOT_R = True
OUTPUT_DIM_RAW = 32

modelToLoad = sys.argv[1]
nz = int(sys.argv[2])
z_dims = int(sys.argv[3])
out_width = int(sys.argv[4])
out_height = int(sys.argv[5])

COPY_KEY = int(sys.argv[6])
t_c, l_c, b_c, r_c = [int(bnd) for bnd in sys.argv[7].split(",")]
PASTE_KEY = int(sys.argv[8])
t_p, l_p, b_p, r_p = [int(bnd) for bnd in sys.argv[9].split(",")]
CONTEXT_KEYS = [int(ck) for ck in sys.argv[10].split(",")]
context_bounds = [int(bnd) for bnd in sys.argv[11].split(",")]
W_ITER = int(sys.argv[12])
C_SIZE = int(sys.argv[13])

print("Copy bounds", t_c, l_c, b_c, r_c)
print("Paste bounds", t_p, l_p, b_p, r_p)
print("Context bounds", context_bounds)

latent_path = sys.argv[14]

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

#print("Loading given generator...")

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
#print("Creating C matrix from randomly sampled k...")
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
copy_z_arr = []
paste_z_arr = []
context_z_arr = []
with open(latent_path, 'r') as latent_input:
    counter = 0

    for line in latent_input:
        list_level = json.loads(line)
        z_elem = list_latent_to_tensor(list_level)
        #print(z_elem.shape)
        if counter == COPY_KEY:
            copy_z_arr.append(z_elem)
        if counter == PASTE_KEY:
            paste_z_arr.append(z_elem)
        if counter in CONTEXT_KEYS:
            context_z_arr.append(z_elem)
        counter += 1

copy_z = torch.Tensor(1, copy_z_arr[0].shape[1], copy_z_arr[0].shape[2], copy_z_arr[0].shape[3])
torch.cat(copy_z_arr, out=copy_z)

paste_z = torch.Tensor(1, paste_z_arr[0].shape[1], paste_z_arr[0].shape[2], paste_z_arr[0].shape[3])
torch.cat(paste_z_arr, out=paste_z)

context_z = torch.Tensor(1, context_z_arr[0].shape[1], context_z_arr[0].shape[2], context_z_arr[0].shape[3])
torch.cat(context_z_arr, out=context_z)

copy_k = context_model(copy_z).detach()
copy_k.requires_grad = False
copy_v = target_model(copy_k).detach()

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
import cp_utils

d_arr = []
d_og_arr = []
zca_arr = []
for i in range(context_k.shape[0]):
    k_key = context_k[i][None,:]

    if COPY_MODE == 'full':
        k_context_mask = torch.ones((1, 1, OUTPUT_DIM_RAW, OUTPUT_DIM_RAW), dtype=torch.float32)
    elif COPY_MODE == 'half':
        k_context_mask = torch.zeros((1, 1, OUTPUT_DIM_RAW, OUTPUT_DIM_RAW), dtype=torch.float32)
        for mi in range(out_height//2):
            for mj in range(out_width):
                k_context_mask[0, 0, mi, mj] = 1
        #k_context_mask = torch.ones((1, 1, out_height, out_width), dtype=torch.float32)
    else:
        k_context_mask = torch.zeros((1, 1, OUTPUT_DIM_RAW, OUTPUT_DIM_RAW), dtype=torch.float32)

        for mi in range(out_height):
            for mj in range(out_width):
                if mi >= context_bounds[i*4] and mi <= context_bounds[i*4+2] and mj >= context_bounds[i*4+1] and mj <= context_bounds[i*4+3]:
                    k_context_mask[0, 0, mi, mj] = 1

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

#print("Calculate v* of given key...")

if COPY_MODE == 'full':
    copy_mask = torch.zeros((1, 1, OUTPUT_DIM_RAW, OUTPUT_DIM_RAW), dtype=torch.float32)
    paste_mask = torch.zeros((1, 1, OUTPUT_DIM_RAW, OUTPUT_DIM_RAW), dtype=torch.float32)
    for mi in range(out_height):
        for mj in range(out_width):
            copy_mask[0, 0, mi, mj] = 1
            paste_mask[0, 0, mi, mj] = 1
elif COPY_MODE == 'half':
    copy_mask = torch.zeros((1, 1, OUTPUT_DIM_RAW, OUTPUT_DIM_RAW), dtype=torch.float32)
    paste_mask = torch.zeros((1, 1, OUTPUT_DIM_RAW, OUTPUT_DIM_RAW), dtype=torch.float32)
    for mi in range(out_height//2):
        for mj in range(out_width):
            copy_mask[0, 0, mi, mj] = 1
            paste_mask[0, 0, mi, mj] = 1
else:
    copy_mask = torch.zeros((1, 1, OUTPUT_DIM_RAW, OUTPUT_DIM_RAW), dtype=torch.float32)
    paste_mask = torch.zeros((1, 1, OUTPUT_DIM_RAW, OUTPUT_DIM_RAW), dtype=torch.float32)


    for mi in range(out_height):
        for mj in range(out_width):
            if mi >= t_c and mi <= b_c and mj >= l_c and mj <= r_c:
                copy_mask[0, 0, mi, mj] = 1
            if mi >= t_p and mi <= b_p and mj >= l_p and mj <= r_p:
                paste_mask[0, 0, mi, mj] = 1

if COPY_MODE == 'inplace':
    interpolated_k_paste_mask = cp_utils.extract_interpolated_mask(copy_mask, copy_k.shape[2:])
else:
    interpolated_k_paste_mask = cp_utils.extract_interpolated_mask(paste_mask, paste_k.shape[2:])

interpolated_v_copy_mask = cp_utils.extract_interpolated_mask(copy_mask, copy_v.shape[2:])
tcv, lcv, bcv, rcv = cp_utils.positive_bounding_box(interpolated_v_copy_mask[0,0])
print("Copy V bounding box:", tcv, lcv, bcv, rcv)
clip_mask = interpolated_v_copy_mask[:, :, tcv:bcv, lcv:rcv]
clip = copy_v[:, :, tcv:bcv, lcv:rcv]
#print("clip", clip.shape)

interpolated_v_paste_mask = cp_utils.extract_interpolated_mask(paste_mask, paste_v.shape[2:])
tpv, lpv, bpv, rpv = cp_utils.positive_bounding_box(interpolated_v_paste_mask[0,0])
print("Paste V bounding box:", tcv, lcv, bcv, rcv)
center = (tpv + bpv) // 2, (lpv + rpv) // 2


ttv, ltv = (max(0, min(e - s, c - s // 2))
            for s, c, e in zip(clip.shape[2:], center, paste_v.shape[2:]))
btv, rtv = ttv + clip.shape[2], ltv + clip.shape[3]


source_k = paste_k

target_v = paste_v.clone()
target_v[:, :, ttv:btv, ltv:rtv] = (1 - clip_mask) * target_v[:, :, ttv:btv, ltv:rtv] + clip_mask * clip
#target_v[:, :, ttv:btv, ltv:rtv] = clip

vr, hr = [ts // ss for ts, ss in zip(target_v.shape[2:], source_k.shape[2:])]
st, sl, sb, sr = ttv // vr, ltv // hr, -(-btv // vr), -(-rtv // hr)
tt, tl, tb, tr = st * vr, sl * hr, sb * vr, sr * hr
print("Goal in bounding box:", st, sl, sb, sr)
print("Goal out bounding box:", tt, tl, tb, tr)
cs, ct = source_k[:, :, st:sb, sl:sr], target_v[:, :, tt:tb, tl:tr]
#print("cs", cs.shape)
#print("ct", ct.shape)

goal_in = cs
goal_out = ct

#if COPY_MODE == 'inplace':
#    target_v, (tb, lb, bb, rb) = cp_utils.move_copy_v_to_paste_center(copy_mask, copy_mask, copy_v, paste_v)
#else:
#    target_v, (tb, lb, bb, rb) = cp_utils.move_copy_v_to_paste_center(copy_mask, paste_mask, copy_v, paste_v)

#print("GOAL OUT BOUNDS", (tb, lb, bb, rb))

#goal_in = torch.mul(paste_k, interpolated_k_paste_mask)
#goal_in = target_k[:, :, tbk:bbk+1, lbk:rbk+1]

#if COPY_MODE == 'inplace':
#    goal_out = cp_utils.move_copy_v_to_paste_center(copy_mask, copy_mask, copy_v, paste_v)
#else:
#    goal_out = cp_utils.move_copy_v_to_paste_center(copy_mask, paste_mask, copy_v, paste_v)

#goal_out = target_v[:, :, tb:bb+1, lb:rb+1]

#print("Merging directions into tensors...")

#all_values = v_new
all_context = torch.cat([di for di in d_arr])
all_zca = torch.cat(zca_arr)

print("ZCA", all_zca.shape)

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
    #print(q.shape)
    top_e_vec = q[:, :DRANK]
    row_dirs = zca_whitened_query_key(zca_matrix, top_e_vec.t())
    #print(row_dirs.shape)
    just_avg = (all_zca).sum(0)
    q, r = torch.qr(row_dirs.permute(1, 0))
    #print(q.shape)
    signs = (q * just_avg[:, None]).sum(0).sign()
    q = q * signs[None, :]
    #print(q.shape)
    final_context = q.permute(1, 0)
else:
    print("no rank reduction")
    final_context = all_context

#print("final_context", final_context.shape)
#sys.exit(0)


#print("Calculating new weights using (k*,v*) pairs...")

print("Goal in:", goal_in.shape)
print("Goal out:", goal_out.shape)
print("Context:", final_context.shape)

#weight = og_insert()
#weight = linear_insert(model, target_model, paste_k, target_v, all_directions, W_ITER, plot=PLOT_R)
#linear_insert_fixed(target_model, goal_in, goal_out, final_context, W_ITER, plot=PLOT_R, lr=LR)
loss_begin, loss_end = normal_insert_fixed(target_model, goal_in, goal_out, final_context, W_ITER=W_ITER, P_ITER=4, lr=LR)
#print(weight.shape)
print("Loss change:", loss_begin, '->', loss_end)

#print("Rewriting model...")

model.main[target_layer].register_parameter('weight', nn.Parameter(target_model[0].weight))
torch.save(model.state_dict(), output_gen_path)

# print("DONE!")