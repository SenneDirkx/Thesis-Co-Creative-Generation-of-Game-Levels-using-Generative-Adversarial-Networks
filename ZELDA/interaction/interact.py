import torch
import torch.nn as nn
import torch.nn.functional as F
from model_generator import GeneratorConv, gumbel_softmax
import numpy as np
from utils import subsequence, make_loader, RunningSecondMoment
import copy
from torch.utils.data import TensorDataset
import matplotlib.pyplot as plt

KEY_ID = 0

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
#z = torch.randn(1, nz, 4, 4)
z = torch.load(f'./levels/generated/{KEY_ID}/latent.pt')
k = context_model(z)
#print("k* shape", k.shape)
v = target_model(k)
#print("v shape", v.shape)
v_flat = v.view(-1)
x = gumbel_softmax(rendering_model(v), temperature)

#k_summed = k.sum(3).sum(2)
#print("k summed shape", k_summed.shape)

W0 = target_model[0].weight
print("weights", W0.shape)
W0_flat = W0.reshape(64, -1).permute(1, 0)

print("Calculating directions...")
k_flat = k.permute(1, 0, 2, 3).reshape(-1, 64).permute(1, 0)
d = torch.matmul(c_inverse, k_flat)
dT = torch.transpose(d, 0, 1)


print("Find v* of given keys...")
x_edited = torch.load(f'./levels/edited/{KEY_ID}/tensor.pt')

criterion = nn.MSELoss()
def compute_loss_v(x_cur):
    return criterion(x_cur,x_edited)

#v_new = torch.randn(1, 16, 13, 10)
with torch.no_grad():
    v_new = torch.clone(v)
v_og = torch.clone(v_new)
v_new.requires_grad = True
v_opt = torch.optim.Adam([v_new], 0.01)
v_losses = []
for it in range(30000):
    with torch.enable_grad():
        x_current = gumbel_softmax(rendering_model(v_new), temperature)
        loss = compute_loss_v(x_current)
        v_opt.zero_grad()
        loss.backward()
        v_losses.append(loss.detach())
        v_opt.step()

v_sum = criterion(v_og, v_new)
x_diff_begin = criterion(gumbel_softmax(rendering_model(v_og), temperature),x_edited)
x_diff_opt = criterion(gumbel_softmax(rendering_model(v_new), temperature), x_edited)
#print('v_sum', v_sum)
#print('x_diff_begin', x_diff_begin)
#print('x_diff_opt', x_diff_opt)

#plt.figure()
#plt.title("V losses")
#plt.plot(v_losses)
#plt.show()


def projected_conv(weight, direction):
    #cosine_map = torch.einsum('oiyx, di -> odyx', weight, direction)
    weight_flat = weight.reshape(64, -1).permute(1,0)
    cosine_map = torch.matmul(weight_flat, direction)
    print(cosine_map.shape)
    #result = torch.einsum('odyx, di -> oiyx', cosine_map, direction)
    result = torch.matmul(cosine_map, direction.permute(1,0)).permute(1, 0).reshape(64, 16, 3, 2)

    return result

f = copy.deepcopy(target_model)

def compute_loss():
    return torch.nn.functional.l1_loss(v_new,f(k))

# set up optimizer
weight = [p for n, p in f.named_parameters()
                if 'weight' in n][0]
params = [weight]
print(params[0].shape)

optimizer = torch.optim.Adam(params, lr=0.01)
losses = []
for it in range(10000):
    with torch.enable_grad():
        loss = compute_loss()
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        print("OG",weight.grad.sum(3).sum(2).sum(1).sum(0))
        print("PROJ",projected_conv(weight.grad, d).sum(3).sum(2).sum(1).sum(0))
        weight.grad[...] = projected_conv(weight.grad, d)
        losses.append(loss.detach())
        optimizer.step()

plt.figure()
plt.title("Weights losses")
plt.plot(losses)
plt.show()

model.main[target_layer].weight = weight
torch.save(model.state_dict(), './rewritten_generator.pth')