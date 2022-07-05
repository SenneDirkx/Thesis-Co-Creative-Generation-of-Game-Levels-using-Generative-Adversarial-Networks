import torch.nn as nn
import torch
from torch.nn import functional as F

def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape)
    return -torch.log(-torch.log(U + eps) + eps)

def gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(logits.size())
    return F.softmax(y / temperature, dim=-1)

def gumbel_softmax(logits, temperature, hard=False):
    """
    ST-gumple-softmax
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """
    permuted_logits = logits.permute(0,2,3,1)
    permuted_y = gumbel_softmax_sample(permuted_logits, temperature)
    #if not hard:
    #    return y.view(-1, nz * categorical_dim)
    
    shape = permuted_y.size()
    _, ind = permuted_y.max(dim=-1)
    y_hard = torch.zeros_like(permuted_y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    # Set gradients w.r.t. y_hard gradients w.r.t. y
    y_hard = (y_hard - permuted_y).detach() + permuted_y
    y_final = y_hard.permute(0,3,1,2)
    return y_final#.view(-1, latent_dim * categorical_dim)

class GeneratorConv(nn.Module):
    def __init__(self, nz, temp):
        super(GeneratorConv, self).__init__()
        self.nz = nz
        self.main = nn.Sequential(
            nn.ConvTranspose2d(self.nz, 64, (3,2), 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(64, 16, (3,2), 2, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(16, 10, (4,2), 1, bias=False),
            #nn.Tanh()
        )

    def forward(self, x, temp, hard):
        x = self.main(x)
        x = gumbel_softmax(x, temp, hard)
        return x

class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)

class GeneratorConv2(nn.Module):
    def __init__(self, nz, temp):
        super(GeneratorConv2, self).__init__()
        self.nz = nz
        n_nodes = 256*3*2
        self.main = nn.Sequential(
            nn.Linear(nz, n_nodes),
            nn.LeakyReLU(0.05),
            Reshape(-1, 256, 3, 2),
            nn.ConvTranspose2d(256, 64, 3, 2, bias=False),
            #nn.BatchNorm2d(64),
            #nn.ReLU(True),
            nn.LeakyReLU(0.05),
            
            nn.ConvTranspose2d(64, 10, (4,3), 2, bias=False),
            #nn.BatchNorm2d(32),
            #nn.ReLU(True),
            #nn.LeakyReLU(0.2),
            
            #nn.ConvTranspose2d(32, 16, (2,2), 2, bias=False),
            #nn.BatchNorm2d(16),
            #nn.ReLU(True),
            #nn.LeakyReLU(0.2),
            
            #nn.ConvTranspose2d(16, 10, (2,1), 2, bias=False),
            #nn.Tanh()
        )

    def forward(self, x, temp, hard):
        x = self.main(x)
        y = gumbel_softmax(x, temp, hard)
        z = F.softmax(x, dim=1)
        return y, z