''' Visualisation tools for analysis '''
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import json
import ops
import os
import utils


# ----------------- Visualisation tools during training ---------------------
def visualise_routers_behaviours(model, data_loader,
                                 no_classes = 10,
                                 objects = ('0','1','2','3','4','5','6','7','8','9'),
                                 fig_scale=None, title='', 
                                 title_font = 20, subtitle_font = 20, axis_font=14,
                                 cuda_on=False,
                                 plot_on = True,
                                 save_as = ''):
    """
    Visualise the probability of reachine a leaf node for different classes.

    Args:
        node (int) : node index. This function gets the list of all the peripheral nodes when the given
                     node is added to the tree, and computest the class probabilities.
        model (nn.Module) : your tree model
        dataloader (data loader):
    Return:

    """
    if cuda_on:
        model.cuda()
    else:
        model.cpu()

    # get the list of edge nodes on respective levels:
    tree_struct = model.tree_struct
    edge_nodes = []
    e_n = 0
    max_level = 0
    while e_n >=0:
        edge_nodes.append(e_n)
        max_level += 1 
        e_n = find_edgenode(tree_struct, max_level)

    # set up the figure size and dimension:
    num_rows = 2*len(edge_nodes) # first row for showing the class fistribution
    num_cols = len(ops.get_past_leaf_nodes(tree_struct, edge_nodes[-1])) # get the list of leaf nodes
    if fig_scale == None:
        fig = plt.figure(figsize=(num_cols,num_rows))
    else:
        fig = plt.figure(figsize=(fig_scale*num_cols, fig_scale*num_rows))
    plt.suptitle(title, fontsize= title_font)

    # -------------- compute and plot stuff ------------------------------
    for level, node in enumerate(edge_nodes):
        print('Computuing histograms for level {}/{}'.format(level, len(edge_nodes) -1)) 
        # compute stuff: 
        y_list, p_list =[], []
        for x, y in data_loader:
            x, y = Variable(x, volatile=True), Variable(y)
            if cuda_on:
                x, y = x.cuda(), y.cuda()
            p, nodes_list = model.compute_routing_probabilities_uptonode(x, node)

            if cuda_on:
                p, y = p.cpu(), y.cpu()

            p_list.append(p.data.numpy())
            y_list.append(y.data.numpy())

        # compute class-specific probabilities for reaching a peripheral node
        c_list = list(range(no_classes)) # [0,1,2,3,4,5,6,7,8,9] # class list
        y_full = np.concatenate(y_list)
        p_full = np.concatenate(p_list) # N x number of peripheral nodes

        node_class_probs = []
        for c in c_list :
            leaf_c = p_full[y_full==c].mean(axis=0)
            node_class_probs.append(leaf_c)

        # C x number of peripheral nodes
        node_class_probs = np.vstack(node_class_probs)

        # Bar chart for node-wise class distributions
        # average probabilitiy of images from a specific class routed to each node
        y_pos = np.arange(len(objects))
        for i, node_idx in enumerate(nodes_list):
            performance = node_class_probs[:, i]
            # print(num_rows, num_cols, 2*num_cols*level+i+1, num_cols*(2*level+1)+i+1 )

            ax1 = fig.add_subplot(num_rows, num_cols, 2*num_cols*level+i+1)
            ax1.bar(y_pos, performance, align='center', alpha=0.5, color='r')
            plt.xticks(y_pos, objects, rotation='vertical', fontsize=axis_font)
            plt.ylim((0,1))
            if i==0:
                ax1.set_ylabel("reaching prob. per class", fontsize=axis_font)
            ax1.set_title('Node '+ str(node_idx), fontsize=subtitle_font)

        # Histogram of reaching probabilities for respective peripheral nodes:
        for i, node_idx in enumerate(nodes_list):
            ax1 = fig.add_subplot(num_rows, num_cols, num_cols*(2*level+1) +i+1)
            ax1.hist(p_full[:, i], normed=False, bins=25, range=(0, 1.0))
            if i==0:
                ax1.set_ylabel("histogram of \n reaching prob. dist.", fontsize=axis_font)

    plt.subplots_adjust(wspace=0.25, hspace=0.25)
    
    if plot_on:
        plt.show()
    
    if save_as:
        # Save the full figure
        print('Save the histogram of splitting as ' + save_as)
        fig.savefig(save_as, format='png', dpi=300)


# ---------------------------- Visualisation tools ----------------------------
# visualise kernels:
def plot_kernels(tensor, num_cols=6, ylabel = '', figsize=None):
    if not tensor.ndim==4:
        raise Exception("assumes a 4D tensor")
    # if not tensor.shape[-1]==3:
    #     raise Exception("last dim needs to be 3 to plot")
    num_kernels = tensor.shape[0]
    num_rows = 1+ num_kernels // num_cols
    if figsize == None:
        fig = plt.figure(figsize=(num_cols,num_rows))
    else:
        fig = plt.figure(figsize=figsize)

    for i in range(tensor.shape[0]):
        img = np.squeeze(tensor[i])
        ax1 = fig.add_subplot(num_rows, num_cols, i+1)
        ax1.imshow(img)
        # ax1.axis('off')
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
        ax1.set_ylabel(ylabel, fontsize=14)

    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.show()

# EXAMPLE USAGE:
# tree = torch.load('~/path/where/tree/model.pth/is/stored')
# for i, node_module in enumerate(tree.tree_modules):
#     tensor = node_module.router.conv1.weight.data.numpy()
#     node_meta = tree.tree_struct[i]
#     ylabel = "Level {}, Node {} ".format(node_meta['level'], node_meta['index'])
#     if node_meta['visited']:
#         plot_kernels(tensor, num_cols=1, ylabel=ylabel, figsize=(3,5))


def visualise_class_distribution_levelwise(level, tree_struct,
                                           model, data_loader,
                                           no_classes = 10,
                                           objects = ('0','1','2','3','4','5','6','7','8','9'),
                                           fig_scale=None, 
                                           title_font = 20, subtitle_font = 20, axis_font=14,
                                           cuda_on=False):
    node_idx = find_edgenode(tree_struct, level)
    visualise_class_distributions_uptonode(node_idx, model, data_loader,
                                           no_classes = no_classes,
                                           objects = objects,
                                           fig_scale = fig_scale, title='level' + str(level),
                                           title_font = title_font, 
                                           subtitle_font = subtitle_font, 
                                           axis_font=axis_font,
                                           cuda_on = cuda_on)


def find_num_nodes_level(tree_struct, level):
    ''' Find the number of nodes in a given level'''
    idx = 0
    for node in tree_struct:
        if node['level']==level :
            idx += 1
    return idx


def find_edgenode(tree_struct, level):
    ''' Given a tree struct, find edge node on the specified level.
    If a given level not found, return None. 
    '''
    idx = 0
    for node in tree_struct:
        if node['level']==level and node['index']> idx:
            idx = node['index']
    if level>0 and idx == 0:
        print('WARNING: specified level not found in the tree ')
        return -1
    else:
        return idx


# visualise histograms of classes and reaching probabilities:
def visualise_class_distributions_uptonode(current_node, model, data_loader,
                                           no_classes = 10,
                                           objects = ('0','1','2','3','4','5','6','7','8','9'),
                                           fig_scale=None, title='', 
                                           title_font = 20, subtitle_font = 20, axis_font=14,
                                           cuda_on=False):
    """
    Visualise the probability of reachine a leaf node for different classes.

    Args:
        node (int): node index. This function gets the list of all the peripheral nodes when the given
                     node is added to the tree, and computest the class probabilities.
        model (nn.Module): your tree model
        dataloader (data loader):

    Examples:
        from torch.utils.data import DataLoader
        from torchvision import datasets, transforms

        # load a tree
        tree = torch.load('~/path/where/tree/model.pth/is/stored')
        # define test data loader
        kwargs = {}
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=False,
            transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])),
                                                  batch_size=1000, shuffle=True, **kwargs)
        # visualise
        visualise_class_distributions(2, tree, test_loader, fig_scale=5, title="Layer 1")
        visualise_class_distributions(5, tree, test_loader, fig_scale=5, title="Layer 2")
        visualise_class_distributions(14, tree, test_loader, fig_scale=5, title="Layer 3")
    """
    if cuda_on:
        model.cuda()
    else:
        model.cpu()

    # assert current_node != 0  # the node cannot be the root

    y_list, p_list =[], []
    for x, y in data_loader:
        x, y = Variable(x, volatile=True), Variable(y)
        if cuda_on:
            x, y = x.cuda(), y.cuda()
        p, nodes_list = model.compute_routing_probabilities_uptonode(x, current_node)

        if cuda_on:
            p, y = p.cpu(), y.cpu()

        p_list.append(p.data.numpy())
        y_list.append(y.data.numpy())

    # compute class-specific probabilities for reaching a peripheral node
    c_list = list(range(no_classes))  # [0,1,2,3,4,5,6,7,8,9] # class list
    y_full = np.concatenate(y_list)
    p_full = np.concatenate(p_list)  # N x number of peripheral nodes

    node_class_probs = []
    for c in c_list :
        leaf_c = p_full[y_full==c].mean(axis=0)
        node_class_probs.append(leaf_c)

    node_class_probs = np.vstack(node_class_probs)  # C x number of peripheral nodes

    # -------------- plot stuff ------------------------------
    num_rows = 2 # first row for showing the class fistribution
    num_cols = p_full.shape[1] # number of perihperal nodes
    if fig_scale == None:
        fig = plt.figure(figsize=(num_cols,num_rows))
    else:
        fig = plt.figure(figsize=(fig_scale*num_cols, fig_scale*num_rows))
    plt.suptitle(title, fontsize= title_font)

    # Bar chart for node-wise class distributions
    # average probabilitiy of images from a specific class routed to each node
    y_pos = np.arange(len(objects))
    for i, node_idx in enumerate(nodes_list):
        performance = node_class_probs[:, i]
        ax1 = fig.add_subplot(num_rows, num_cols,i+1)
        ax1.bar(y_pos, performance, align='center', alpha=0.5, color='r')
        plt.xticks(y_pos, objects)
        plt.ylim((0,1))
        if i==0:
            ax1.set_ylabel("reaching prob. per class", fontsize=axis_font)
        ax1.set_title('Node '+ str(node_idx), fontsize=subtitle_font)

    # Histogram of reaching probabilities for respective peripheral nodes:
    for i, node_idx in enumerate(nodes_list):
        ax1 = fig.add_subplot(num_rows, num_cols,num_cols+i+1)
        ax1.hist(p_full[:, i], normed=False, bins=25, range=(0, 1.0))
        if i==0:
            ax1.set_ylabel("histogram of \n reaching prob. dist.", fontsize=axis_font)

    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.show()

# plot train/test loss for a single model
def print_performance(jasonfile, model_name='model_1', figsize=(5,5)) :
    """ Inspect performance of a single model
    """
    records = json.load(open(jasonfile, 'r'))
    print('\n'+model_name)
    print("        train_best_loss: {}".format(records['train_best_loss']))
    print("        valid_best_loss: {}".format(records['valid_best_loss']))
    print("        test_best_loss: {}".format(records['test_best_loss']))
    
    # Plot train/test loss
    fig = plt.figure(figsize=figsize)
    plt.plot(np.arange(len(records['test_epoch_loss'])), np.array(records['test_epoch_loss']),
             linestyle='-.', color='b', label='test epoch loss')     
    plt.plot(np.arange(len(records['train_epoch_loss']), dtype=float), np.array(records['train_epoch_loss']), 
             color='r', linestyle='-', label='train epoch loss')
    plt.legend(loc='upper right')
    plt.ylabel('epoch wise loss (average CE loss)')
    plt.xlabel('epoch number')


def plot_performance(jasonfiles, model_names=[], ymin=0.0, ymax=1.0, figsize=(5,5), title='', finetune_position=False) :
    """ Plot train/test loss for multiple models.
    """
    # TODO: currently only supports up to 8 models at a time due to color types
    fig = plt.figure(figsize=figsize)
    color = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']

    if not(model_names):
        model_names = [str(i) for i in range(len(jasonfiles))]

    for i, f in enumerate(jasonfiles):
        # load the information: 
        records = json.load(open(f, 'r'))
        
        # Plot train/test loss
        plt.plot(np.arange(len(records['valid_epoch_loss'])), np.array(records['valid_epoch_loss']),
                 color=color[i], linestyle='-.', label='valid epoch loss: ' + model_names[i])
        plt.plot(np.arange(len(records['train_epoch_loss']), dtype=float), np.array(records['train_epoch_loss']), 
                 color=color[i], linestyle='-', label='train epoch loss: '  + model_names[i])

        if finetune_position:
            plt.axvline(
                x=len(records['valid_epoch_loss']) - records['epochs_finetune'],
                color=color[i], linestyle='--',
            )

    plt.ylim(ymax=ymax, ymin=ymin)
    plt.ylabel('Cross-entropy loss', fontsize=15)
    plt.xlabel('Epoch', fontsize=15)
    plt.legend(loc='upper right', fontsize=13)
    plt.title(title)


def plot_accuracy(
        jasonfiles, model_names=[],
        figsize=(5,5), ymin=0.0, ymax=100.0, title='',
        finetune_position=False, color=None):
    """Plot test accuracy (%) for multiple models on the same axis. """
    fig = plt.figure(figsize=figsize)
    if not(color):
        color = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']

    if not(model_names):
        model_names = [str(i) for i in range(len(jasonfiles))]

    for i, f in enumerate(jasonfiles):
        # load the information: 
        records = json.load(open(f, 'r'))
        
        # Plot train/test loss
        plt.plot(np.arange(len(records['test_epoch_accuracy']), dtype=float), np.array(records['test_epoch_accuracy']), 
                 color=color[i], linestyle='-', label=model_names[i])

        plt.ylabel('Test accuracy (%)', fontsize=15)
        plt.xlabel('Epoch', fontsize=15)
        plt.ylim(ymax=ymax, ymin=ymin)

        # choose est based on validation accuracy
        best_epoch_acc = records['valid_epoch_accuracy'].index(max(records['valid_epoch_accuracy']))
        print(model_names[i] + ': test accuracy = {}'.format(records['test_epoch_accuracy'][best_epoch_acc]))
        if finetune_position:
            plt.axvline(x=len(records['valid_epoch_loss'])-records['epochs_finetune'], color=color[i], linestyle='--')
    plt.legend(loc='lower right', fontsize=13)
    plt.title(title)


def compute_accuracy(jasonfiles, model_names=[], name =''):
    if not(model_names):
        model_names = [str(i) for i in range(len(jasonfiles))]
    accuracy_1 = []
    accuracy_2 = []
    print('\nPerformance: '+ name)
    for i, f in enumerate(jasonfiles):
        # load the information: 
        records = json.load(open(f, 'r'))
        
        # choose the best epoch based on validation loss:
        best_epoch_loss = records['valid_epoch_loss'].index(min(records['valid_epoch_loss']))
        best_epoch_acc = records['valid_epoch_accuracy'].index(max(records['valid_epoch_accuracy']))
        print(model_names[i] + ':test1 = {}, test2 = {}'.format(records['test_epoch_accuracy'][best_epoch_loss], records['test_epoch_accuracy'][best_epoch_acc]))
        accuracy_1.append(records['test_epoch_accuracy'][best_epoch_loss])
        accuracy_2.append(records['test_epoch_accuracy'][best_epoch_acc])
    return accuracy_1, accuracy_2


# load a model and measure peformance on a heldout dataset: 
def compute_error(model_file, data_loader, cuda_on=False, name = ''):
    ''' 
    Args:
        model_file (str): model parameters 
        data_dataloader (torch.utils.data.DataLoader): data loader
    ''' 
    # load the model
    model = torch.load(model_file)
    if cuda_on:
        model.cuda()

    # compute the error 
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in data_loader:
        if cuda_on:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).data[
            0]  # sum up batch loss
        pred = output.data.max(1, keepdim=True)[
            1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(data_loader.dataset)
    print(name + 'Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(data_loader.dataset),
        100. * correct / len(data_loader.dataset)))


def compute_number_of_params(
        model_files, model_names,
        print_on=True, is_gpu=True, include_routers=True,
):
    n_total_list = []
    n_max_list = []
    n_min_list = []
    n_avr_list = []

    map_location = None
    if not(is_gpu):
        map_location = lambda storage, loc: storage
    
    for f, name in zip(model_files, model_names):
        net = torch.load(f, map_location=map_location)
        n_total, n_max, n_min, n_avg = utils.get_number_of_params_summary(
            net, name='Model: ' + name,
            print_on=print_on, include_routers=include_routers,
        )
        n_total_list.append(n_total)
        n_max_list.append(n_max)
        n_min_list.append(n_min)
        n_avr_list.append(n_avg)
    return n_total_list, n_max_list, n_min_list, n_avr_list


def visualise_treestructures(fig_dir, figsize=(5,5), fig_name=''):
    figs_list = [f for f in os.listdir(fig_dir)]
    figs_list.sort()

    # load and plot
    plt.figure(figsize=figsize)

    if not(fig_name):
        fig_name = figs_list[-1]

    img = mpimg.imread(fig_dir + fig_name)
    print('     Plotting:  ' + fig_name)
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
