import torch.nn as nn
import torch.nn.functional as F
import torch
import copy
import matplotlib.pyplot as plt
from interaction_utils.general import set_requires_grad

def estimate_v(rendering_model, start_v, x_edited, out_width, out_height, V_ITER=1000, plot=False):
    # maybe also l1 loss??
    #print("render output", rendering_model(start_v)[:, :, :out_height, :out_width])
    #print("edited", x_edited)

    criterion = nn.MSELoss()
    def compute_loss_v(x_cur):
        return criterion(x_cur,x_edited)

    #v_new = torch.randn(1, 16, 13, 10)
    with torch.no_grad():
        v_new = torch.clone(start_v)
    #v_og = torch.clone(v_new)
    v_new.requires_grad = True
    v_opt = torch.optim.Adam([v_new], lr=0.001)
    v_losses = []
    for _ in range(V_ITER):
        with torch.enable_grad():
            x_current = rendering_model(v_new)
            x_current = x_current[:, :, :out_height, :out_width]
            #a = x_current.argmax(axis = 1)
            #x_onehotted = torch.zeros(x_current.shape).scatter(1, a.unsqueeze(1), 1.0)
            
            ret = F.gumbel_softmax(x_current, tau=1, hard=True, dim=1)
            #print(ret.shape)
            #print(ret[0,:,10,10])
            #print(x_current.shape)
            loss = compute_loss_v(ret)
            v_opt.zero_grad()
            loss.backward()
            v_losses.append(loss.detach())
            v_opt.step()

    #v_sum = criterion(v_og, v_new)
    #x_diff_begin = criterion(gumbel_softmax(rendering_model(v_og), temperature),x_edited)
    #x_diff_opt = criterion(gumbel_softmax(rendering_model(v_new), temperature), x_edited)
    if plot:
        plt.figure()
        plt.title("V losses")
        plt.plot(v_losses)
        plt.show()

    v_new.requires_grad = False
    return v_new


def og_insert(target_model, k, v_new, W_ITER=1000):
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

def linear_insert(model, target_model, keys, vals, d_og, W_ITER=1000, plot=False):
    set_requires_grad(False, model)
    key, val = keys.detach(), vals.detach()
    key.requires_grad = False
    val.requires_grad = False
    original_weight = [p for n, p in target_model.named_parameters()
                    if 'weight' in n][0]
    hooked_module = [module for module in target_model.modules()
                        if getattr(module, 'weight', None) is original_weight][0]
    del hooked_module._parameters['weight']
    original_weight = original_weight[None, :].permute(0, 2, 1, 3, 4)
    ws = original_weight.shape
    print(ws)
    lambda_param = torch.zeros(
        ws[0], ws[1], d_og.shape[0],
        ws[3], ws[4], device=original_weight.device,
        requires_grad=True)
    print(lambda_param.shape)
    print(d_og.shape)
    old_forward = hooked_module.forward

    def new_forward(x):
        # weight_1 = weight_0 + Lambda D
        #print("Weight shape", original_weight.shape)
        #print("lambda param shape", lambda_param.shape)
        #print("direction shape", d.shape)
        to_be_added = torch.einsum('godyx, di... -> goiyx', lambda_param, d_og)
        #to_be_added = lambda_param
        #print("einsum shape", to_be_added.shape)
        #hooked_module.weight = (original_weight + to_be_added).reshape(16, 64, 3, 2).permute(1, 0, 2, 3)
        hooked_module.weight = (original_weight + to_be_added).reshape(ws[1], ws[2], ws[3], ws[4]).permute(1, 0, 2, 3)
        result = old_forward(x)
        return result
    hooked_module.forward = new_forward

    # when computing the loss, hook the weights to be modified by Lambda D
    def compute_loss():
        loss = torch.nn.functional.l1_loss(val, target_model(key))
        return loss

    # run the optimizer
    params = [lambda_param]
    optimizer = torch.optim.Adam(params, lr=0.00001)
    losses = []
    counter = 0
    for _ in range(W_ITER):
        with torch.enable_grad():
            loss = compute_loss()
            optimizer.zero_grad()
            loss.backward()
            if counter == 0:
                print("loss:", loss.detach())
                counter = 10
            losses.append(loss.detach())
            optimizer.step()
            counter -= 1
    
    if plot:
        plt.figure()
        plt.title("Weights losses")
        plt.plot(losses)
        plt.show()

    with torch.no_grad():
        # OK now fill in the learned weights and undo the hook.
        print("Weight shape", original_weight.shape)
        print("lambda param shape", lambda_param.shape)
        print("direction shape", d_og.shape)
        new_weight = original_weight + torch.einsum('godyx, di... -> goiyx', lambda_param, d_og)
        print("new weight shape", new_weight.shape)
        original_weight[...] = new_weight
        del hooked_module.weight
        #hooked_module.weight = original_weight.reshape(16, 64, 3, 2).permute(1, 0, 2, 3)
        #hooked_module.register_parameter('weight', nn.Parameter(original_weight.reshape(16, 64, 3, 2).permute(1, 0, 2, 3)))
        hooked_module.forward = old_forward
    
    return original_weight.reshape(ws[1], ws[2], ws[3], ws[4]).permute(1, 0, 2, 3)

def linear_insert_fixed(target_model, key, val, context, W_ITER=1000, plot=False, lr=0.001):
    
    key, val = [d.detach() for d in [key, val]]
    original_weight = [p for n, p in target_model.named_parameters() if 'weight' in n][0]
    hooked_module = [module for module in target_model.modules() if getattr(module, 'weight', None) is original_weight][0]
    del hooked_module._parameters['weight']
    ws = original_weight.shape
    print("ws", ws)
    lambda_param = torch.zeros(
        ws[1], context.shape[0],
        ws[2], ws[3], device=original_weight.device,
        requires_grad=True)
    print("ls", lambda_param.shape)
    print("cs", context.shape)
    old_forward = hooked_module.forward

    def new_forward(x):
        # weight_1 = weight_0 + Lambda D
        new_weight = torch.einsum('odyx, di -> oiyx', lambda_param, context).permute(1,0,2,3)
        #print("nws", new_weight.shape)
        hooked_module.weight = (original_weight + new_weight)
        result = old_forward(x)
        return result
    hooked_module.forward = new_forward

    # when computing the loss, hook the weights to be modified by Lambda D
    def compute_loss():
        loss = torch.nn.functional.l1_loss(val, target_model(key))
        return loss

    # run the optimizer
    params = [lambda_param]
    optimizer = torch.optim.Adam(params, lr=lr)
    for it in range(W_ITER):
        with torch.enable_grad():
            loss = compute_loss()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(it, loss)
            if it == W_ITER-1:
                print("LOSS", loss)
    with torch.no_grad():
        # OK now fill in the learned weights and undo the hook.
        new_weight = torch.einsum('odyx, di -> oiyx', lambda_param, context).permute(1,0,2,3)

        original_weight[...] = (original_weight + new_weight)
        del hooked_module.weight
        hooked_module.register_parameter('weight', original_weight)
        hooked_module.forward = old_forward


def normal_insert_fixed(target_model, key, val, context, W_ITER=1000, P_ITER=10, lr=0.05):
    # print('inserting keys %s' % list(key.keys()))
    key, val = [d.detach() for d in [key, val]]

    LOW_RANK_INSERT = True

    def compute_loss():
        return torch.nn.functional.l1_loss(val, target_model(key))
    # set up optimizer
    weight = [p for n, p in target_model.named_parameters() if 'weight' in n][0]
    params = [weight]
    if LOW_RANK_INSERT:
        # The assumption now is that context is orthonormal.
        with torch.no_grad():
            ortho_weight = weight - projected_conv(weight, context)
    optimizer = torch.optim.Adam(params, lr=lr)

    for it in range(W_ITER):
        with torch.enable_grad():
            loss = compute_loss()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(it, loss)
            # Project to rank-one over context direction
            if LOW_RANK_INSERT and (it % P_ITER == 0 or it == W_ITER - 1):
                with torch.no_grad():
                    weight[...] = (
                        ortho_weight + projected_conv(weight, context))

def projected_conv(weight, direction):
    cosine_map = torch.einsum('oiyx, di -> odyx', weight.permute(1,0,2,3), direction)
    result = torch.einsum('odyx, di -> oiyx', cosine_map, direction).permute(1,0,2,3)
    return result