import torch
from model_generator import GeneratorConv, gumbel_softmax
import numpy as np
from utils import subsequence, make_loader, RunningSecondMoment, call_compute
import copy
from torch.utils.data import TensorDataset

nz = 128
temperature = 1.0

generator = GeneratorConv(nz, temperature)
generator.load_state_dict(torch.load("./generator.pth"))
generator.eval()

firstlayer = '3'
lastlayer = '3'

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

with torch.no_grad():
    z_sample = zds[0][0][None]
    k_sample = context_model(z_sample)
    v_sample = target_model(k_sample)
    x_sample = gumbel_softmax(rendering_model(v_sample), temperature)

    def separate_key_reps(zbatch):
        acts = context_model(zbatch)
        sep_pix = acts.permute(0, 2, 3, 1).reshape(-1, acts.shape[1])
        print(sep_pix.shape)
        return sep_pix
    
    loader = make_loader(zds, None, 10)
    r2mom = RunningSecondMoment()
    for batch in loader:
        sample = call_compute(separate_key_reps, batch)
        r2mom.add(sample)
    
    c_matrix = r2mom.moment()
print('c_matrix', c_matrix.shape)


z = torch.randn(1, nz, 4, 4)
k = context_model(z)
print(k.shape)
v = target_model(k)
x = gumbel_softmax(rendering_model(v), temperature)

d = torch.matmul(c_matrix.inverse(), k)
print(d.shape)
#torch.save(result, f'./levels/generated/{i}/tensor.pt')

