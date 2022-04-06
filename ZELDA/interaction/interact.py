import torch
import torch.nn as nn
import torch.nn.functional as F
from model_generator import GeneratorConv, gumbel_softmax
import numpy as np
from utils import subsequence, make_loader, RunningSecondMoment, set_requires_grad
import copy
from torch.utils.data import TensorDataset
import matplotlib.pyplot as plt

KEY_ID = 0
V_ITER = 1000
W_ITER = 1000

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
k_flat = k.permute(1, 0, 2, 3).reshape(-1, 64).permute(1, 0)
d = torch.matmul(c_inverse, k_flat).detach()
d.requires_grad = False
dT = torch.transpose(d, 0, 1)

d_og = d.reshape(64, 6, 5)[None, :]


print("Find v* of given keys...")
x_edited = torch.load(f'./levels/edited/{KEY_ID}/tensor.pt')

# maybe also l1 loss??
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
for it in range(V_ITER):
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

v_new.requires_grad = False
keys = [k]
vals = [v_new]

#print('v_sum', v_sum)
#print('x_diff_begin', x_diff_begin)
#print('x_diff_opt', x_diff_opt)

#plt.figure()
#plt.title("V losses")
#plt.plot(v_losses)
#plt.show()

print("Calculating new weights using (k*,v*) pairs...")

def og_insert():
    def projected_conv(weight, direction):
        #cosine_map = torch.einsum('oiyx, di -> odyx', weight, direction)
        weight_flat = weight.reshape(64, -1).permute(1,0)
        cosine_map = torch.matmul(weight_flat, direction)
        #result = torch.einsum('odyx, di -> oiyx', cosine_map, direction)
        result = torch.matmul(cosine_map, direction.permute(1,0)).permute(1, 0).reshape(64, 16, 3, 2)

        return result

    f = copy.deepcopy(target_model)
    f.requires_grad = False

    def compute_loss_weigths():
        return torch.nn.functional.l1_loss(v_new,f(k))

    # set up optimizer
    weight = [p for n, p in f.named_parameters()
                    if 'weight' in n][0]
    params = [weight]
    print(params[0].shape)

    optimizer = torch.optim.Adam(params, lr=0.01)
    losses = []
    for _ in range(W_ITER):
        with torch.enable_grad():
            loss = compute_loss_weigths()
            optimizer.zero_grad()
            loss.backward()
            #print("OG",weight.grad.sum(3).sum(2).sum(1).sum(0))
            #print("PROJ",projected_conv(weight.grad, d).sum(3).sum(2).sum(1).sum(0))
            weight.grad[...] = projected_conv(weight.grad, d)
            losses.append(loss.detach())
            optimizer.step()

    plt.figure()
    plt.title("Weights losses")
    plt.plot(losses)
    plt.show()
    return weight

def linear_insert():
    set_requires_grad(False, model)
    key, val = keys[KEY_ID].detach(), vals[KEY_ID].detach()
    key.requires_grad = False
    val.requires_grad = False
    original_weight = [p for n, p in target_model.named_parameters()
                    if 'weight' in n][0]
    hooked_module = [module for module in target_model.modules()
                        if getattr(module, 'weight', None) is original_weight][0]
    del hooked_module._parameters['weight']
    original_weight = original_weight[None, :].permute(0, 2, 1, 3, 4)
    ws = original_weight.shape
    lambda_param = torch.zeros(
        ws[0], ws[1], d.shape[0],
        ws[3], ws[4], device=original_weight.device,
        requires_grad=True)
    old_forward = hooked_module.forward

    def new_forward(x):
        # weight_1 = weight_0 + Lambda D
        #print("Weight shape", original_weight.shape)
        #print("lambda param shape", lambda_param.shape)
        #print("direction shape", d.shape)
        to_be_added = torch.einsum('godyx, di... -> goiyx', lambda_param, d_og)
        #print("einsum shape", to_be_added.shape)
        hooked_module.weight = (original_weight + to_be_added).reshape(16, 64, 3, 2).permute(1, 0, 2, 3)
        result = old_forward(x)
        return result
    hooked_module.forward = new_forward

    # when computing the loss, hook the weights to be modified by Lambda D
    def compute_loss():
        loss = torch.nn.functional.l1_loss(val, target_model(key))
        return loss

    # run the optimizer
    params = [lambda_param]
    optimizer = torch.optim.Adam(params, lr=0.01)
    for _ in range(W_ITER):
        with torch.enable_grad():
            loss = compute_loss()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    with torch.no_grad():
        # OK now fill in the learned weights and undo the hook.
        print("Weight shape", original_weight.shape)
        print("lambda param shape", lambda_param.shape)
        print("direction shape", d.shape)
        new_weight = original_weight + torch.einsum('godyx, di... -> goiyx', lambda_param, d_og)
        print("new weight shape", new_weight.shape)
        original_weight[...] = new_weight
        del hooked_module.weight
        #hooked_module.weight = original_weight.reshape(16, 64, 3, 2).permute(1, 0, 2, 3)
        #hooked_module.register_parameter('weight', nn.Parameter(original_weight.reshape(16, 64, 3, 2).permute(1, 0, 2, 3)))
        hooked_module.forward = old_forward
    
    return original_weight.reshape(16, 64, 3, 2).permute(1, 0, 2, 3)

#weight = og_insert()
weight = linear_insert()
#print(weight.shape)

print("Rewriting model...")
model.main[target_layer].register_parameter('weight', nn.Parameter(weight))
torch.save(model.state_dict(), './rewritten_generator.pth')

print("DONE!")