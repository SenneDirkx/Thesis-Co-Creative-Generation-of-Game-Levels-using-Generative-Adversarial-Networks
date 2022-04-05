import hiddenlayer as hl
from model_generator import GeneratorConv
import torch

batch_size = 16
nz = 128 # latent vector size
temperature = 1.0
model = GeneratorConv(nz, temperature)
batch = torch.randn(1, nz, 4, 4)

transforms = [ hl.transforms.Prune('Constant'), hl.transforms.Fold("Shape > Gather", 
            "SG", "SG"),hl.transforms.Fold("SG > Unsqueeze", 
            "SGU", "SGU"),
            hl.transforms.Fold("Concat > ConstantOfShape > RandomUniformLike > Add > Log > Neg > Add > Log > Neg", 
            "G1", "G1"),
            hl.transforms.Prune('SGU'),
            hl.transforms.Prune('G1'),
            hl.transforms.Prune('Cast'),
            hl.transforms.Fold("Transpose > Add > Div > Softmax", 
            "GumbelSoftmax", "GumbelSoftmax"),
            hl.transforms.Prune('SG'),
            hl.transforms.Prune('Shape'),
            hl.transforms.Prune('ArgMax'),
            hl.transforms.Prune('Sub'),
            hl.transforms.Prune('Unsqueeze'),
            hl.transforms.Prune('Concat'),
            hl.transforms.Prune('Reshape'),
            hl.transforms.Prune('ConstantOfShape'),
            hl.transforms.Prune('Expand'),
            hl.transforms.Prune('Scatter'),
            hl.transforms.Prune('Add'),
            hl.transforms.Prune('Transpose'),
            #hl.transforms.Fold("(Transpose > Add) | ((Tranpose > ((Shape > Gather > Unsqueeze) | (Shape > Gather > Unsqueeze) | (Shape > Gather > Unsqueeze) | (Shape > Gather > Unsqueeze)) > Concat > ConstantOfShape > RandomUniformLike > Add > Log > Neg > Add > Log > Neg) > Add)", 
            #"G2", "G2") ] # Removes Constant nodes from graph.
]
graph = hl.build_graph(model, args=(batch, temperature, False), transforms=transforms)
graph.theme = hl.graph.THEMES['blue'].copy()
graph.save('rnn_hiddenlayer', format='png')