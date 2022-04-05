import torch
import copy
from collections import OrderedDict
from torch.utils.data.sampler import Sampler
import numpy as np

def subsequence(sequential, first_layer=None, last_layer=None,
                after_layer=None, upto_layer=None, single_layer=None,
                share_weights=False):
    '''
    Creates a subsequence of a pytorch Sequential model, copying over
    modules together with parameters for the subsequence.  Only
    modules from first_layer to last_layer (inclusive) are included,
    or modules between after_layer and upto_layer (exclusive).
    Handles descent into dotted layer names as long as all references
    are within nested Sequential models.
    If share_weights is True, then references the original modules
    and their parameters without copying them.  Otherwise, by default,
    makes a separate brand-new copy.
    '''
    assert ((single_layer is None) or (first_layer is last_layer is after_layer is upto_layer is None))
    if single_layer is not None:
        first_layer = single_layer
        last_layer = single_layer
    first, last, after, upto = [None if d is None else d.split('.')
                                for d in [first_layer, last_layer, after_layer, upto_layer]]
    return hierarchical_subsequence(sequential, first=first, last=last,
                                    after=after, upto=upto, share_weights=share_weights)


def hierarchical_subsequence(sequential, first, last, after, upto,
                             share_weights=False, depth=0):
    '''
    Recursive helper for subsequence() to support descent into dotted
    layer names.  In this helper, first, last, after, and upto are
    arrays of names resulting from splitting on dots.  Can only
    descend into nested Sequentials.
    '''
    assert (last is None) or (upto is None)
    assert (first is None) or (after is None)
    if first is last is after is upto is None:
        return sequential if share_weights else copy.deepcopy(sequential)
    assert isinstance(sequential, torch.nn.Sequential), ('.'.join(
        (first or last or after or upto)[:depth] or 'arg') + ' not Sequential')
    including_children = (first is None) and (after is None)
    included_children = OrderedDict()
    (F, FN), (L, LN), (A, AN), (U, UN) = [
        (d[depth], (None if len(d) == depth + 1 else d))
        if d is not None else (None, None)
        for d in [first, last, after, upto]]
    for name, layer in sequential._modules.items():
        if name == F:
            first = None
            including_children = True
        if name == A and AN is not None:
            after = None
            including_children = True
        if name == U and UN is None:
            upto = None
            including_children = False
        if including_children:
            FR, LR, AR, UR = [n if n is None or n[depth] == name else None
                              for n in [FN, LN, AN, UN]]
            chosen = hierarchical_subsequence(layer,
                                              first=FR, last=LR, after=AR, upto=UR,
                                              share_weights=share_weights, depth=depth + 1)
            if chosen is not None:
                included_children[name] = chosen
        if name == L:
            last = None
            including_children = False
        if name == U and UN is not None:
            upto = None
            including_children = False
        if name == A and AN is None:
            after = None
            including_children = True
    for name in [first, last, after, upto]:
        if name is not None:
            raise ValueError('Layer %s not found' % '.'.join(name))
    # Omit empty subsequences except at the outermost level,
    # where we should not return None.
    if not len(included_children) and depth > 0:
        return None
    return torch.nn.Sequential(included_children)

def make_loader(dataset, sample_size=None, batch_size=10, sampler=None,
                **kwargs):
    '''Utility for creating a dataloader on fixed sample subset.'''
    if isinstance(dataset, torch.Tensor):
        dataset = torch.utils.data.TensorDataset(dataset)
    if sampler is None:
        if sample_size is not None:
            if sample_size > len(dataset):
                #pbar.print("Warning: sample size %d > dataset size %d" %
                #           (sample_size, len(dataset)))
                sample_size = len(dataset)
            sampler = FixedSubsetSampler(list(range(sample_size)))
    return torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        **kwargs)

class FixedSubsetSampler(Sampler):
    """Represents a fixed sequence of data set indices.
    Subsets can be created by specifying a subset of output indexes.
    """

    def __init__(self, samples):
        self.samples = samples

    def __iter__(self):
        return iter(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, key):
        return self.samples[key]

    def subset(self, new_subset):
        return FixedSubsetSampler(self.dereference(new_subset))

    def dereference(self, indices):
        '''
        Translate output sample indices (small numbers indexing the sample)
        to input sample indices (larger number indexing the original full set)
        '''
        return [self.samples[i] for i in indices]

def resolve_state_dict(s):
    if isinstance(s, str):
        return np.load(s, allow_pickle=True)
    return s

def progress_addbmm(accum, x, y, batch_size):
    '''
    Break up very large adbmm operations into batches so progress can be seen.
    '''
    if x.shape[0] <= batch_size:
        return accum.addbmm_(x, y)
    for i in range(0, x.shape[0], batch_size):
        accum.addbmm_(x[i:i + batch_size], y[i:i + batch_size])
    return accum

class RunningSecondMoment:
    '''
    Running computation. Use this when the entire non-centered 2nd-moment
    "covariance-like" matrix is needed, and when the whole matrix fits
    in the GPU.
    '''

    def __init__(self, state=None):
        if state is not None:
            self.set_state_dict(resolve_state_dict(state))
            return
        self.count = 0
        self.mom2 = None

    def add(self, a):
        if len(a.shape) == 1:
            a = a[None, :]
        # Initial batch reveals the shape of the data.
        if self.count == 0:
            self.mom2 = a.new(a.shape[1], a.shape[1]).zero_()
        batch_count = a.shape[0]
        # If more than 10 billion operations, divide into batches.
        sub_batch = -(-(10 << 30) // (a.shape[1] * a.shape[1]))
        # Update the covariance using the batch deviation
        self.count += batch_count
        #print(a.shape)
        #print(a[:, :, None].shape)
        #print(a)
        #print(a[:, :, None])
        progress_addbmm(self.mom2, a[:, :, None], a[:, None, :], sub_batch)

    def cpu_(self):
        self.mom2 = self.mom2.cpu()

    def cuda_(self):
        self.mom2 = self.mom2.cuda()

    def to_(self, device):
        self.mom2 = self.mom2.to(device)

    def moment(self):
        return self.mom2 / self.count

    def state_dict(self):
        return dict(
            constructor=self.__module__ + '.' +
            self.__class__.__name__ + '()',
            count=self.count,
            mom2=self.mom2.cpu().numpy())

    def set_state_dict(self, dic):
        self.count = dic['count'].item()
        self.mom2 = torch.from_numpy(dic['mom2'])

