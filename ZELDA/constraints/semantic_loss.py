from pysdd.sdd import Vtree, SddManager
from pathlib import Path
import torch

def import_sdd(name):
    vtree_file = "./" + name + ".vtree"
    vtree = Vtree.from_file(vtree_file)
    manager = SddManager.from_vtree(vtree)
    
    root = manager.read_sdd_file(bytes(Path(__file__).parent / (name + ".sdd")))
    return root

def traverse_tree(root):
    if not root.is_decision():
        if root.is_literal():
            return str(root.literal)
        elif root.is_false():
            return "f"
        elif root.is_true():
            return "t"
        else:
            return "BRUH"
    else:
        elems = []
        for elem in root.elements():
            elems.append("(" + traverse_tree(elem[0]) + " & " + traverse_tree(elem[1]) + ")")
        return " ( " + " | ".join(elems) + " ) "

def analyze_tree_with_vars(root, vars):
    if not root.is_decision():
        if root.is_literal():
            if root.literal < 0:
                return 1 - vars[abs(root.literal)]
            else:
                return vars[root.literal]
        elif root.is_false():
            return 0
        elif root.is_true():
            return 1
        else:
            return "BRUH"
    else:
        sum = 0
        for elem in root.elements():
            sum += analyze_tree_with_vars(elem[0], vars) * analyze_tree_with_vars(elem[1], vars)
        return sum

def create_computation_graph(root, vars):
    shape = vars[0].shape
    if not root.is_decision():
        if root.is_literal():
            if root.literal < 0:
                return torch.ones(shape, requires_grad=True) - vars[abs(root.literal)]
            else:
                return vars[root.literal]
        elif root.is_false():
            return torch.zeros(shape, requires_grad=True)
        elif root.is_true():
            return torch.ones(shape, requires_grad=True)
        else:
            return "BRUH"
    else:
        sum = torch.zeros(shape, requires_grad=True)
        for elem in root.elements():
            sum = torch.add(sum, torch.mul(create_computation_graph(elem[0], vars), create_computation_graph(elem[1], vars)))
        return sum

def constraint_loss(output, target):
    sdd = import_sdd("basic_constraint")
    create_computation_graph(sdd, [door_left, door_right, door_top, door_bottom,])
    #print(traverse_tree(sdd))
    #print(analyze_tree_with_vars(sdd, [0, 1, 1, 1, 1] + [0, 0, 0, 1]))
    print(output.shape)
    door_left = torch.prod(output[:,:, 7:9, 0])
    door_right = torch.prod(output[:,:, 7:9, 10])
    door_top = torch.prod(output[:,:, 0, 4:7])
    door_bottom = torch.prod(output[:,:, 15, 4:7])
    print(door_left)
    print(door_right)
    print(door_top)
    print(door_bottom)


#loss()