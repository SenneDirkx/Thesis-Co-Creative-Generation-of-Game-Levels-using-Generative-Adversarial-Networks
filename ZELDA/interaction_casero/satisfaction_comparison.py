from model_generator import GeneratorConv
import torch
from semantic_loss_pytorch import SemanticLoss
import matplotlib.pyplot as plt
import numpy as np

nz = 128
temperature = 1

og_gen = GeneratorConv(nz, temperature)
og_gen.load_state_dict(torch.load("./generator.pth"))
og_gen.eval()

new_gen = GeneratorConv(nz, temperature)
new_gen.load_state_dict(torch.load("./rewritten_generator.pth"))
new_gen.eval()

og_levels = og_gen(torch.rand(10000, 128, 4, 4), temperature, False)
new_levels = new_gen(torch.rand(10000, 128, 4, 4), temperature, False)

sl = SemanticLoss("../constraints/basic_constraint.sdd", "../constraints/basic_constraint.vtree")

def wall_var(fake_levels):
    b_size = fake_levels.shape[0]
    
    top_outer_wall = torch.prod(fake_levels[:, 8, 0, :].view(b_size, -1), dim=1)
    bottom_outer_wall = torch.prod(fake_levels[:, 8, 15, :].view(b_size, -1), dim=1)
    left_outer_wall = torch.prod(fake_levels[:, 8, 1:15, 0].view(b_size, -1), dim=1)
    right_outer_wall = torch.prod(fake_levels[:, 8, 1:15, 10].view(b_size, -1), dim=1)
    
    outer_walls = top_outer_wall * bottom_outer_wall * left_outer_wall * right_outer_wall
    return outer_walls.unsqueeze(1)
    
def semantic_loss(fake_levels):
    outer_walls = wall_var(fake_levels) + 1e-8
    #print(torch.sum(outer_walls))
    loss = sl(probabilities=outer_walls)
    return loss

data = [semantic_loss(og_levels).item(), semantic_loss(new_levels).item()]

gens = ('Original generator', 'Adapted generator')
y_pos = np.arange(len(gens))

fig, ax = plt.subplots()

hbars = ax.barh(y_pos, data, align='center')
ax.set_yticks(y_pos, labels=gens)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Semantic loss')
ax.set_title('Original generator vs Adapted generator')

# Label with specially formatted floats
ax.bar_label(hbars)
ax.set_xlim(right=0.1)  # adjust xlim to fit labels

plt.show()