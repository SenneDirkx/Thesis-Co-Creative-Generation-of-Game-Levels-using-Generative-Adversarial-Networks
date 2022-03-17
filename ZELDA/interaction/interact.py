import torch
from model_generator import GeneratorConv

nz = 128
temperature = 1.0

generator = GeneratorConv(nz, temperature)
generator.load_state_dict(torch.load("./generator.pth"))
generator.eval()

activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

generator.main[2].register_forward_hook(get_activation('relu1'))
generator.main[3].register_forward_hook(get_activation('conv_transpose2'))


I = torch.randn(10, nz, 4, 4)
O = generator(I, temperature, False)
K = torch.flatten(activation['relu1'], start_dim=1).transpose(0, 1)
V = torch.flatten(activation['conv_transpose2'], start_dim=1).transpose(0, 1)

print("I", I.shape)
print("K", K.shape)
print("V", V.shape)
print("O", O.shape)

C = torch.matmul(K, K.transpose(0, 1))

print("C", C.shape)

#print(C)
print(torch.diag(C))

i = torch.randn(1, nz, 4, 4)
o = generator(i, temperature, False)
k = torch.flatten(activation['relu1'], start_dim=1).transpose(0, 1)
v = torch.flatten(activation['conv_transpose2'], start_dim=1).transpose(0, 1)

d = torch.matmul(C.inverse(), k)
#torch.save(result, f'./levels/generated/{i}/tensor.pt')

