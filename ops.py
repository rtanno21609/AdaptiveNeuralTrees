"""Operations"""
import torch
import torch.nn as nn
import torch.utils.data.sampler as sampler
import numpy as np


# --------- Neural network modules -----------------------------
def weight_init(m):
    classname = m.__class__.__name__
    print(classname)
    if classname.find('Conv2d') != -1:
        nn.init.xavier_normal(m.weight, gain=np.sqrt(2))
        if m.bias is not None:
            nn.init.constant(m.bias, 0.0)
    elif classname.find('Linear') != -1:
        nn.init.xavier_uniform(m.weight, gain=np.sqrt(2))
        if m.bias is not None:
            nn.init.constant(m.bias, 0.0)


# --------- straight-through (ST) estimators for binary neurons ---------
# reference: https://r2rt.com/binary-stochastic-neurons-in-tensorflow.html
class ST_Indicator(torch.autograd.Function):
    """ Straight-through indicator function 1(0.5 =< input):
    rounds a tensor whose values are in [0,1] to a tensor with values in {0, 1},
    using identity for its gradient.
    """
    def forward(self, input):
        return torch.round(input)

    def backward(self, grad_output):
        """
        In the backward pass, just use unscaled straight-through estimator.
        """
        return grad_output


class ST_StochasticIndicator(torch.autograd.Function):
    """ Stochastic version of ST_Indicator:
    indicator function 1(z =< input) where z is drawn from Uniform[0,1]
    with identity for its gradient.
    """
    def forward(self, input):
        """
        Args:
            input (float tensor of values between 0 and 1): threshold prob

        Return:
            input: float tensor of values 0 or 1
        """

        # draw samples from Uniform[0,1]
        z = input.new(input.size()).uniform_(0, 1)
        # z = torch.FloatTensor(input.shape).uniform_(0, 1)
        return torch.abs(torch.round(input-z+0.5))

    def backward(self, grad_output):
        """
        In the backward pass, just use unscaled straight-through estimator.
        """
        return grad_output


# ---------------------- Loss functions -----------------------------
def neg_ce_fairflip(p, coeff):
    ''' Compute the negative CE between p and Bernouli(0.5)
    '''
    nce = - 0.5*coeff*(torch.log(p + 1e-10) + torch.log(1.0 - p + 1e-10)).mean()
    return nce


def weighted_cross_entropy(input, target, weights):
    """ Compute cross entropy weighted per example
    got most ideas from https://github.com/pytorch/pytorch/issues/563

    Args:
        input (torch tensor): (N,C) log probabilities e.g. F.log_softmax(y_hat)
        target (torch tensor): (N) where each value is 0 <= targes[i] <=C-1
        weights (torch tensor): (N) per-example weight

    Return:
        nlhd: negative loglijelihood
    """
    # compute (N,) tensor of the log probabilities of the correct classes
    logpy = torch.gather(input, 1, target.view(-1,1)).view(-1)

    # compute the negative log-likelihood loss:
    nlhd = -(logpy * weights).mean()
    # nlhd = -logpy.mean()

    return nlhd


def differential_entropy(input, bins=10, min=0.0, max=1.0):
    """ Approximate dfferential entropy: H(x) = E[-log P(x)]
    Fit a histogram and compute the maximum likelihood estimator of  entropy
    i.e. H^{hat}(x) = - \sum_{i} P(x_i) log(P(x_i))

    See https://en.wikipedia.org/wiki/Entropy_estimation

     Args:
        input (torch tensor): (N) probabilities

    """

    p_x_list = torch.histc(input, bins=bins, min=min, max=max)/input.size(0)
    h = (-p_x_list*torch.log(p_x_list+1e-8)).sum()
    return h


def coefficient_of_variation(input):
    """ Compute the coefficient of varation std/mean:
    """
    epsilon = 1e-10
    return input.std()/(input.mean()+epsilon)


# ---------------------- Tree operations --------------------
def count_number_transforms(node_idx, tree_struct):
    """ Get the number of transforms up to and including node_idx 
    """
    nodes, _ = get_path_to_root(node_idx, tree_struct)
    count = 0
    for i in nodes:
        if tree_struct[i]['transformed']:
            count += 1
    return count


def count_number_transforms_after_last_downsample(node_idx, tree_struct):
    nodes, _ = get_path_to_root(node_idx, tree_struct)
    
    # get the last node at which features are downsampled
    last_idx=0
    for i in nodes:
        if tree_struct[i]['transformed'] and tree_struct[i]['downsampled']:
            last_idx = i
            
    # count the number of transformations
    count = 0
    for i in nodes[last_idx:]:
        if tree_struct[i]['transformed'] and not(tree_struct[i]['downsampled']):
            count += 1
        
    return count


def get_leaf_nodes(struct):
    """ Get the list of leaf nodes.
    """
    leaf_list = []
    for idx, node in enumerate(struct):
        if node['is_leaf']:
            leaf_list.append(idx)
    return leaf_list


def get_past_leaf_nodes(struct, current_idx):
    """ Get the list of nodes that were leaves when the specified node is added
    to the tree.
    """
    # if node_current is the root node.
    if current_idx == 0:
        return [0]

    # otherwise:
    leaf_list = [current_idx]
    node_current = struct[current_idx]

    # get the brother node:
    parent_idx = struct[current_idx]['parent']
    node_r = struct[parent_idx]['right_child']
    node_l = struct[parent_idx]['left_child']   
    if current_idx==node_r:
        bro_idx = node_l
    else:
        bro_idx = node_r

    if not(struct[parent_idx]['extended']): # if its parent is extended, it has not brother
        leaf_list.append(bro_idx)

    # get the list of leaf nodes whose the indices are below the parent node
    leaf_nodes_below_parent\
        = [node['index'] for node in struct
           if node['index'] < parent_idx and node['is_leaf']]
    leaf_list = leaf_list + leaf_nodes_below_parent

    # get the list of nodes on the same level and before node_current:
    leaf_same_level = [node['index'] for node in struct
                       if node['index'] < current_idx
                       and node['index']!= bro_idx
                       and node['level'] == node_current['level']]
    leaf_list = leaf_list + leaf_same_level

    # get the list of nodes on the level above and after the parent node
    leaf_above_level = [node['index'] for node in struct
                        if node['index'] > parent_idx
                        and node['level'] == node_current['level']-1]
    leaf_list = leaf_list + leaf_above_level

    # sort:
    leaf_list.sort()
    return leaf_list


def get_path_to_root_old(node_idx, struct):
    """ Get the list of nodes from the current node to the root.
    [0, n1,....., node_idx]
    """
    paths_list = []
    while node_idx >= 0:
        paths_list.append(node_idx)
        node_idx = get_parent(node_idx, struct)

    return paths_list[::-1]


def get_path_to_root(node_idx, struct):
    """ Get two lists:
    First, list of all nodes from the root node to the given node
    Second, list of left-child-status (boolean) of each edge between nodes in the first list
    """
    paths_list = []
    left_child_status = []
    while node_idx >= 0:
        if node_idx > 0:  # ignore parent node
            lcs = get_left_or_right(node_idx, struct)
            left_child_status.append(lcs)
        paths_list.append(node_idx)
        node_idx = get_parent(node_idx, struct)

    paths_list = paths_list[::-1]
    left_child_status = left_child_status[::-1]
    return paths_list, left_child_status


def get_parent(node_idx, struct):
    """ Get index of parent node
    """
    return struct[node_idx]['parent']


def get_left_or_right(node_idx, struct):
    """ Return True if the node is a left child of its parent.
    o/w return false.
    """
    parent_node = struct[node_idx]['parent']
    return struct[parent_node]['left_child'] == node_idx


def node_pred(nodes, edges, tree_modules, input):
    """ Perform prediction on a given node given its path on the tree.
    e.g.
    nodes = [0, 1, 4, 10]
    edges = [True, False, False]
    """
    # Transform data and compute probability of reaching the last node in path
    prob = 1.0
    for node, state in zip(nodes[:-1], edges):
        input = tree_modules[node]['transform'](input)
        prob *= tree_modules[node]['router'](input) if state \
            else (1.0 - tree_modules[node]['router'](input))

    node_final = nodes[-1]
    input = tree_modules[node_final]['transform'](input)

    # Perform classification with the last node:
    prob = torch.unsqueeze(prob, 1)
    y_pred = prob * tree_modules[node_final]['LR'](input)
    return y_pred


def node_pred_split(input, nodes, edges, tree_modules, node_left, node_right):
    """ Perform prediction on a split node given its path on the tree.
    Here, the last node in the  list "nodes" is assumed to be split.
    e.g.
    nodes = [0, 1, 4, 10]
    edges = [True, False, False]
    then, node 10 is assumed to be split.

    Args:
        input (torch.Variable): input images
        nodes (list): list of all nodes (index) on the path between root and given node
        edges (list): list of left-child-status (boolean) of each edge between nodes in
                      the list 'nodes'
        tree_modules (list): list of all node modules in the tree

        node_left (dict) : candidate node for the left child of node
        node_right (dict): candidate node for the right child of node
    Return:
    """
    # Transform data and compute probability of reaching the last node in path
    prob = 1.0
    for node, state in zip(nodes[:-1], edges):
        input = tree_modules[node]['transform'](input)
        prob *= tree_modules[node]['router'](input) if state \
            else (1.0 - tree_modules[node]['router'](input))

    node_final = nodes[-1]
    input = tree_modules[node_final]['transform'](input)
    prob = torch.unsqueeze(prob, 1)

    # Perform classification with the last node:
    prob_last = tree_modules[node_final]['router'](input) if state \
        else (1.0 - tree_modules[node_final]['router'](input))
    prob_last = torch.unsqueeze(prob_last, 1)

    # Split the last node:
    y_pred = prob * (prob_last * node_left['LR'](node_left['transform'](input))
                     + (1.0 - prob_last) * node_right['LR'](
        node_right['transform'](input))
                     )
    return y_pred


def get_params_node(grow, node_idx, model):
    """Get the list of trainable parameters at the given node.

    If grow=True, then fetch the local parameters
    (i.e. parent router + 2 children transformers and solvers)
    """
    if grow:
        names = [name for name, param in model.named_parameters()
                 if ('.'+str(node_idx)+'.router' in name) 
                 or ('child' in name and not('router' in name))]

        print("\nSelectively optimising the parameters below: ")
        for name in names: print('          '+name)

        params = [param for name, param in model.named_parameters()
                  if ('.'+str(node_idx)+'.router' in name) 
                  or ('child' in name and not('router' in name))]

        # set the parameters to require gradients
        for p in params:
            p.requires_grad = True

        return params, names
    else:
        print("\nPerform global training")
        return list(model.parameters()), [name for name, param in
                                          model.named_parameters()]


# ---------------------- Data loader extra options ----------------------------
class ChunkSampler(sampler. Sampler):
    """ Samples elements sequentially from some offset.
    Args:
        num_samples: # of desired datapoints
        start: offset where we should start selecting from

    Source:
        https://github.com/pytorch/vision/issues/168

    Examples:
        NUM_TRAIN = 49000
        NUM_VAL = 1000
        cifar10_train = dset.CIFAR10('./cs231n/datasets', train=True, download=True,
                            transform=T.ToTensor())
        loader_train = DataLoader(cifar10_train, batch_size=64, sampler=ChunkSampler(NUM_TRAIN, 0))

        cifar10_val = dset.CIFAR10('./cs231n/datasets', train=True, download=True, transform=T.ToTensor())
        loader_val = DataLoader(cifar10_val, batch_size=64, sampler=ChunkSampler(NUM_VAL, NUM_TRAIN))

    """
    def __init__(self, num_samples, start=0, shuffle=False):
        self.num_samples = num_samples
        self.start = start
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            return iter((torch.randperm(self.num_samples) + self.start).long())
        else:
            return iter(range(self.start, self.start + self.num_samples))

    def __len__(self):
        return self.num_samples
