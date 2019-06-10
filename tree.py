"""
Main script for training an adaptive neural tree (ANT).
"""
from __future__ import print_function
import argparse
import os
import sys
import json
import time
import numpy as np

import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch import optim

import matplotlib
matplotlib.use('agg')

from data import get_dataloaders, get_dataset_details
from models import Tree, One
from ops import get_params_node
from utils import define_node, get_scheduler, set_random_seed
from visualisation import visualise_routers_behaviours


# Experiment settings
parser = argparse.ArgumentParser(description='Adaptive Neural Trees')
parser.add_argument('--experiment', '-e', dest='experiment', default='tree', help='experiment name')
parser.add_argument('--subexperiment','-sube', dest='subexperiment', default='', help='experiment name')

parser.add_argument('--dataset', default='mnist', help='dataset type')
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
parser.add_argument('--gpu', type=str, default="", help='which GPU to use')
parser.add_argument('--seed', type=int, default=0, metavar='S', help='random seed')
parser.add_argument('--num_workers', type=int, default=0, metavar='N', help='number of threads for data-loader')

# Optimization settings:
parser.add_argument('--batch-size', type=int, default=256, metavar='N', help='input batch size for training')
parser.add_argument('--log-interval', type=int, default=10, metavar='N', help='how many batches to wait before logging training status')
parser.add_argument('--augmentation_on', action='store_true', default=False, help='perform data augmentation')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate')
parser.add_argument('--scheduler', type=str, default="", help='learning rate scheduler')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M', help='SGD momentum')
parser.add_argument('--valid_ratio', '-vr', dest='valid_ratio', type=float, default=0.1, metavar='LR', help='validation set ratio')

parser.add_argument('--criteria', default='avg_valid_loss', help='growth criteria')
parser.add_argument('--epochs_node', type=int, default=50, metavar='N', help='max number of epochs to train per node during the growth phase')
parser.add_argument('--epochs_finetune', type=int, default=100, metavar='N', help='number of epochs for the refinement phase')
parser.add_argument('--epochs_patience', type=int, default=5, metavar='N', help='number of epochs to be waited without improvement at each node during the growth phase')
parser.add_argument('--maxdepth', type=int, default=10, help='maximum depth of tree')
parser.add_argument('--finetune_during_growth', action='store_true', default=False, help='refine the tree globally during the growth phase')
parser.add_argument('--epochs_finetune_node', type=int, default=1, metavar='N', help='number of epochs to perform global refinement at each node during the growth phase')


# Solver, router and transformer modules:
parser.add_argument('--router_ver', '-r_ver', dest='router_ver', type=int, default=1, help='default router version')
parser.add_argument('--router_ngf', '-r_ngf', dest='router_ngf', type=int, default=1, help='number of feature maps in routing function')
parser.add_argument('--router_k', '-r_k', dest='router_k', type=int, default=28, help='kernel size in routing function')
parser.add_argument('--router_dropout_prob', '-r_drop', dest='router_dropout_prob', type=float, default=0.0, help='drop-out probabilities for router modules.')

parser.add_argument('--transformer_ver', '-t_ver', dest='transformer_ver', type=int, default=1, help='default transformer version: identity')
parser.add_argument('--transformer_ngf', '-t_ngf', dest='transformer_ngf', type=int, default=3, help='number of feature maps in residual transformer')
parser.add_argument('--transformer_k', '-t_k', dest='transformer_k', type=int, default=5, help='kernel size in transfomer function')
parser.add_argument('--transformer_expansion_rate', '-t_expr', dest='transformer_expansion_rate', type=int, default=1, help='default transformer expansion rate')
parser.add_argument('--transformer_reduction_rate', '-t_redr', dest='transformer_reduction_rate', type=int, default=2, help='default transformer reduction rate')

parser.add_argument('--solver_ver', '-s_ver', dest='solver_ver', type=int, default=1, help='default router version')
parser.add_argument('--solver_inherit', '-s_inh', dest='solver_inherit',  action='store_true', help='inherit the parameters of the solver when defining two new ones for splitting a node')
parser.add_argument('--solver_dropout_prob', '-s_drop', dest='solver_dropout_prob', type=float, default=0.0, help='drop-out probabilities for solver modules.')

parser.add_argument('--downsample_interval', '-ds_int', dest='downsample_interval', type=int, default=0, help='interval between two downsampling operations via transformers i.e. 0 = downsample at every transformer')
parser.add_argument('--batch_norm', '-bn', dest='batch_norm', action='store_true', default=False, help='turn batch norm on')

# Visualisation:
parser.add_argument('--visualise_split', action='store_true', help='visuliase how the test dist is split by the routing function')

args = parser.parse_args()

# GPUs devices:
args.cuda = not args.no_cuda and torch.cuda.is_available()
if args.gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

# Set the seed for repeatability
set_random_seed(args.seed, args.cuda)

# Define a dictionary for post-inspection of the model:
records = vars(args)
records['time'] = 0.0
records['counter'] = 0  # number of optimization steps

records['train_nodes'] = []  # node indices for each logging interval
records['train_loss'] = []   # avg. train. loss for each log interval
records['train_best_loss'] = np.inf  # best train. loss
records['train_epoch_loss'] = []  # epoch wise train loss

records['valid_nodes'] = []
records['valid_best_loss_nodes'] = []
records['valid_best_loss_nodes_split'] = []
records['valid_best_loss_nodes_ext'] = []
records['valid_best_root_nosplit'] = np.inf
records['valid_best_loss'] = np.inf
records['valid_best_accuracy'] = 0.0
records['valid_epoch_loss'] = []
records['valid_epoch_accuracy'] = []

records['test_best_loss'] = np.inf
records['test_best_accuracy'] = 0.0
records['test_epoch_loss'] = []
records['test_epoch_accuracy'] = []


# -----------------------------  Data loaders ---------------------------------
train_loader, valid_loader, test_loader, NUM_TRAIN, NUM_VALID = get_dataloaders(
    args.dataset, args.batch_size, args.augmentation_on,
    cuda=args.cuda, num_workers=args.num_workers,
)
args.input_nc, args.input_width, args.input_height, args.classes = \
    get_dataset_details(args.dataset)
args.no_classes = len(args.classes)


# -----------------------------  Components ----------------------------------
def train(model, data_loader, optimizer, node_idx):
    """ Train step"""
    model.train()
    train_loss = 0
    no_points = 0
    train_epoch_loss = 0

    # train the model
    for batch_idx, (x, y) in enumerate(data_loader):
        optimizer.zero_grad()
        if args.cuda:
            x, y = x.cuda(), y.cuda()
        x, y = Variable(x), Variable(y)
        y_pred, p_out = model(x)

        loss = F.nll_loss(y_pred, y)
        train_epoch_loss += loss.data[0] * y.size(0)
        train_loss += loss.data[0] * y.size(0)
        loss.backward()
        optimizer.step()

        records['counter'] += 1
        no_points += y.size(0)

        if batch_idx % args.log_interval == 0:
            # show the interval-wise average loss:
            train_loss /= no_points
            records['train_loss'].append(train_loss)
            records['train_nodes'].append(node_idx)

            sys.stdout.flush()
            sys.stdout.write('\t      [{}/{} ({:.0f}%)]      Loss: {:.6f} \r'.
                    format(batch_idx*len(x), NUM_TRAIN,
                    100. * batch_idx / NUM_TRAIN, train_loss))

            train_loss = 0
            no_points = 0

    # compute average train loss for the epoch
    train_epoch_loss /= NUM_TRAIN
    records['train_epoch_loss'].append(train_epoch_loss)
    if train_epoch_loss < records['train_best_loss']:
        records['train_best_loss'] = train_epoch_loss

    print('\nTrain set: Average loss: {:.4f}'.format(train_epoch_loss))


def valid(model, data_loader, node_idx, struct):
    """ Validation step """
    model.eval()
    valid_epoch_loss = 0
    correct = 0

    for data, target in data_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)

        # sum up batch loss
        valid_epoch_loss += F.nll_loss(
            output, target, size_average=False,
        ).data[0]

        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    valid_epoch_loss /= NUM_VALID
    valid_epoch_accuracy = 100. * correct / NUM_VALID
    records['valid_epoch_loss'].append(valid_epoch_loss)
    records['valid_epoch_accuracy'].append(valid_epoch_accuracy)

    if valid_epoch_loss < records['valid_best_loss']:
        records['valid_best_loss'] = valid_epoch_loss

    if valid_epoch_accuracy > records['valid_best_accuracy']:
        records['valid_best_accuracy'] = valid_epoch_accuracy

    # see if the current node is root and undergoing the initial training
    # prior to the growth phase.
    is_init_root_train = not model.split and not model.extend and node_idx == 0

    # save the best split model during node-wise training as model_tmp.pth
    if not is_init_root_train and model.split and \
            valid_epoch_loss < records['valid_best_loss_nodes_split'][node_idx]:
        records['valid_best_loss_nodes_split'][node_idx] = valid_epoch_loss
        checkpoint_model('model_tmp.pth', model=model)
        checkpoint_msc(struct, records)

    # save the best extended model during node-wise training as model_ext.pth
    if not is_init_root_train and model.extend and \
            valid_epoch_loss < records['valid_best_loss_nodes_ext'][node_idx]:
        records['valid_best_loss_nodes_ext'][node_idx] = valid_epoch_loss
        checkpoint_model('model_ext.pth', model=model)
        checkpoint_msc(struct, records)

    # separately store best performance for the initial root training
    if is_init_root_train \
            and valid_epoch_loss < records['valid_best_root_nosplit']:
        records['valid_best_root_nosplit'] = valid_epoch_loss
        checkpoint_model('model_tmp.pth', model=model)
        checkpoint_msc(struct, records)

    # saving model during the refinement (fine-tuning) phase
    if not is_init_root_train and \
            valid_epoch_loss < records['valid_best_loss_nodes'][node_idx]:
        records['valid_best_loss_nodes'][node_idx] = valid_epoch_loss
        if not model.split and not model.extend:
            checkpoint_model('model_tmp.pth', model=model)
            checkpoint_msc(struct, records)

    end = time.time()
    records['time'] = end - start
    print(
        'Valid set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'
        '\nTook {} seconds. '.format(
            valid_epoch_loss, correct, NUM_VALID,
            100. * correct / NUM_VALID, records['time'],
        ),
    )
    return valid_epoch_loss


def test(model, data_loader):
    """ Test step """
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in data_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).data[0]
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(data_loader.dataset)
    test_accuracy = 100. * correct / len(data_loader.dataset)
    records['test_epoch_loss'].append(test_loss)
    records['test_epoch_accuracy'].append(test_accuracy)

    if test_loss < records['test_best_loss']:
        records['test_best_loss'] = test_loss

    if test_accuracy > records['test_best_accuracy']:
        records['test_best_accuracy'] = test_accuracy

    end = time.time()
    print(
        'Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'
        '\nTook {} seconds. '.format(
            test_loss, correct, len(data_loader.dataset),
            100. * correct / len(data_loader.dataset), end - start,
        ),
    )


def _load_checkpoint(model_file_name):
    save_dir = "./experiments/{}/{}/{}/{}".format(
        args.dataset, args.experiment, args.subexperiment, 'checkpoints',
    )
    model = torch.load(save_dir + '/' + model_file_name)
    if args.cuda:
        model.cuda()
    return model


def checkpoint_model(model_file_name, struct=None, modules=None, model=None, figname='hist.png', data_loader=None):
    if not(os.path.exists(os.path.join("./experiments", args.dataset, args.experiment, args.subexperiment))):
        os.makedirs(os.path.join("./experiments", args.dataset, args.experiment, args.subexperiment, 'figures'))
        os.makedirs(os.path.join("./experiments", args.dataset, args.experiment, args.subexperiment, 'checkpoints'))
    
    # If model is not given, then build one. 
    if not(model) and modules and struct:
        model = Tree(struct, modules, cuda_on=args.cuda)
        
    # save the model:
    save_dir = "./experiments/{}/{}/{}/{}".format(args.dataset, args.experiment, args.subexperiment, 'checkpoints')
    model_path = save_dir + '/' + model_file_name
    torch.save(model, model_path)
    print("Model saved to {}".format(model_path))

    # save tree histograms:
    if args.visualise_split and not(data_loader is None):
        save_hist_dir = "./experiments/{}/{}/{}/{}".format(args.dataset, args.experiment, args.subexperiment, 'figures')
        visualise_routers_behaviours(model, data_loader, fig_scale=6, axis_font=20, subtitle_font=20, 
                                     cuda_on=args.cuda, objects=args.classes, plot_on=False, 
                                     save_as=save_hist_dir + '/' + figname)


def checkpoint_msc(struct, data_dict):
    """ Save structural information of the model and experimental results.

    Args:
        struct (list) : list of dictionaries each of which contains
            meta information about each node of the tree.
        data_dict (dict) : data about the experiment (e.g. loss, configurations)
    """
    if not(os.path.exists(os.path.join("./experiments", args.dataset, args.experiment, args.subexperiment))):
        os.makedirs(os.path.join("./experiments", args.dataset, args.experiment, args.subexperiment, 'figures'))
        os.makedirs(os.path.join("./experiments", args.dataset, args.experiment, args.subexperiment, 'checkpoints'))

    # save the tree structures as a json file:
    save_dir = "./experiments/{}/{}/{}/{}".format(args.dataset,args.experiment,args.subexperiment,'checkpoints')
    struct_path = save_dir + "/tree_structures.json"
    with open(struct_path, 'w') as f:
        json.dump(struct, f)
    print("Tree structure saved to {}".format(struct_path))

    # save the dictionary as jason file:
    dict_path = save_dir + "/records.json"
    with open(dict_path, 'w') as f_d:
        json.dump(data_dict, f_d)
    print("Other data saved to {}".format(dict_path))


def get_decision(criteria, node_idx, tree_struct):
    """ Define the splitting criteria

    Args:
        criteria (str): Growth criteria.
        node_idx (int): Index of the current node.
        tree_struct (list) : list of dictionaries each of which contains
            meta information about each node of the tree.

    Returns:
        The function returns one of the following strings
            'split': split the node
            'extend': extend the node
            'keep': keep the node as it is
    """
    if criteria == 'always':  # always split or extend
        if tree_struct[node_idx]['valid_accuracy_gain_ext'] > tree_struct[node_idx]['valid_accuracy_gain_split'] > 0.0:
            return 'extend'
        else:
            return 'split'
    elif criteria == 'avg_valid_loss':
        if tree_struct[node_idx]['valid_accuracy_gain_ext'] > tree_struct[node_idx]['valid_accuracy_gain_split'] and \
                        tree_struct[node_idx]['valid_accuracy_gain_ext'] > 0.0:
            print("Average valid loss is reduced by {} ".format(tree_struct[node_idx]['valid_accuracy_gain_ext']))
            return 'extend'

        elif tree_struct[node_idx]['valid_accuracy_gain_split'] > 0.0:
            print("Average valid loss is reduced by {} ".format(tree_struct[node_idx]['valid_accuracy_gain_split']))
            return 'split'

        else:
            print("Average valid loss is aggravated by split/extension."
                  " Keep the node as it is.")
            return 'keep'
    else:
        raise NotImplementedError(
            "specified growth criteria is not available. ",
        )


def optimize_fixed_tree(
        model, tree_struct, train_loader,
        valid_loader, test_loader, no_epochs, node_idx,
):
    """ Train a tree with fixed architecture.

    Args:
        model (torch.nn.module): tree model
        tree_struct (list): list of dictionaries which contain information
                            about all nodes in the tree.
        train_loader (torch.utils.data.DataLoader) : data loader of train data
        valid_loader (torch.utils.data.DataLoader) : data loader of valid data
        test_loader (torch.utils.data.DataLoader) : data loader of test data
        no_epochs (int): number of epochs for training
        node_idx (int): index of the node you want to optimize

    Returns:
        returns the trained model and newly added nodes (if grown).
    """
    # get if the model is growing or fixed
    grow = (model.split or model.extend)

    # define optimizer and trainable parameters
    params, names = get_params_node(grow, node_idx,  model)
    for i, (n, p) in enumerate(model.named_parameters()):
        if not(n in names):
            # print('(Fix)   ' + n)
            p.requires_grad = False
        else:
            # print('(Optimize)     ' + n)
            p.requires_grad = True

    for i, p in enumerate(params):
        if not(p.requires_grad):
            print("(Grad not required)" + names[i])

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, params), lr=args.lr,
    )
    if args.scheduler:
        scheduler = get_scheduler(args.scheduler, optimizer, grow)

    # monitor nodewise best valid loss:
    if not(not(grow) and node_idx==0) and len(records['valid_best_loss_nodes']) == node_idx:
        records['valid_best_loss_nodes'].append(np.inf)
    
    if not(not(grow) and node_idx==0) and len(records['valid_best_loss_nodes_split']) == node_idx:
        records['valid_best_loss_nodes_split'].append(np.inf)

    if not(not(grow) and node_idx==0) and len(records['valid_best_loss_nodes_ext']) == node_idx:
        records['valid_best_loss_nodes_ext'].append(np.inf)

    # start training
    min_improvement = 0.0  # acceptable improvement in loss for early stopping
    valid_loss = np.inf
    patience_cnt = 1

    for epoch in range(1, no_epochs + 1):
        print("\n----- Layer {}, Node {}, Epoch {}/{}, Patience {}/{}---------".
              format(tree_struct[node_idx]['level'], node_idx, 
                     epoch, no_epochs, patience_cnt, args.epochs_patience))
        train(model, train_loader, optimizer, node_idx)
        valid_loss_new = valid(model, valid_loader, node_idx, tree_struct)
        
        # learning rate scheduling:
        if args.scheduler == 'plateau':
            scheduler.step(valid_loss_new)
        elif args.scheduler == 'step_lr':
            scheduler.step()
        
        test(model, test_loader)

        if not((valid_loss-valid_loss_new) > min_improvement) and grow:
            patience_cnt += 1
        valid_loss = valid_loss_new*1.0
        
        if patience_cnt > args.epochs_patience > 0:
            print('Early stopping')
            break
 
    # load the node-wise best model based on validation accuracy:
    if no_epochs > 0 and grow:
        if model.extend:
            print('return the node-wise best extended model')
            model = _load_checkpoint('model_ext.pth')
        else:
            print('return the node-wise best split model')
            model = _load_checkpoint('model_tmp.pth')

    # return the updated models:
    tree_modules = model.update_tree_modules()
    if model.split:
        child_left, child_right = model.update_children()
        return model, tree_modules, child_left, child_right
    elif model.extend:
        child_extension = model.update_children()
        return model, tree_modules, child_extension
    else:
        return model, tree_modules


def grow_ant_nodewise():
    """The main function for optimising an ANT """

    # ############## 0: Define the root node and optimise ###################
    # define the root node:
    tree_struct = []  # stores graph information for each node
    tree_modules = []  # stores modules for each node
    root_meta, root_module = define_node(
        args, node_index=0, level=0, parent_index=-1, tree_struct=tree_struct,
    )
    tree_struct.append(root_meta)
    tree_modules.append(root_module)

    # train classifier on root node (no split no extension):
    model = Tree(
        tree_struct, tree_modules, split=False, extend=False, cuda_on=args.cuda,
    )
    if args.cuda:
        model.cuda()

    # optimise
    model, tree_modules = optimize_fixed_tree(
        model, tree_struct,
        train_loader, valid_loader, test_loader, args.epochs_node, node_idx=0,
    )
    checkpoint_model('model.pth', struct=tree_struct, modules=tree_modules)
    checkpoint_msc(tree_struct, records)

    # ######################## 1: Growth phase starts ########################
    nextind = 1
    last_node = 0
    for lyr in range(args.maxdepth):
        print("---------------------------------------------------------------")
        print("\nAt layer " + str(lyr))
        for node_idx in range(len(tree_struct)):
            change = False
            if tree_struct[node_idx]['is_leaf'] and not(tree_struct[node_idx]['visited']):

                print("\nProcessing node " + str(node_idx))

                # -------------- Define children candidate nodes --------------
                # ---------------------- (1) Split ----------------------------
                # left child
                identity = True
                meta_l, node_l = define_node(
                    args,
                    node_index=nextind, level=lyr+1,
                    parent_index=node_idx, tree_struct=tree_struct,
                    identity=identity,
                )
                # right child
                meta_r, node_r = define_node(
                    args,
                    node_index=nextind+1, level=lyr+1,
                    parent_index=node_idx, tree_struct=tree_struct,
                    identity=identity,
                )
                # inheriting solver modules to facilitate optimization:
                if args.solver_inherit and meta_l['identity'] and meta_r['identity'] and not(node_idx == 0):
                    node_l['classifier'] = tree_modules[node_idx]['classifier']
                    node_r['classifier'] = tree_modules[node_idx]['classifier']

                # define a tree with a new split by adding two children nodes:
                model_split = Tree(tree_struct, tree_modules,
                                   split=True, node_split=node_idx,
                                   child_left=node_l, child_right=node_r,
                                   extend=False,
                                   cuda_on=args.cuda)

                # -------------------- (2) Extend ----------------------------
                # define a tree with node extension
                meta_e, node_e = define_node(
                    args,
                    node_index=nextind,
                    level=lyr+1,
                    parent_index=node_idx,
                    tree_struct=tree_struct,
                    identity=False,
                )
                # Set the router at the current node as one-sided One().
                # TODO: this is not ideal as it changes tree_modules
                tree_modules[node_idx]['router'] = One()

                # define a tree with an extended edge by adding a node
                model_ext = Tree(tree_struct, tree_modules,
                                 split=False,
                                 extend=True, node_extend=node_idx,
                                 child_extension=node_e,
                                 cuda_on=args.cuda)

                # ---------------------- Optimise -----------------------------
                best_tr_loss = records['train_best_loss']
                best_va_loss = records['valid_best_loss']
                best_te_loss = records['test_best_loss']

                print("\n---------- Optimizing a binary split ------------")
                if args.cuda:
                    model_split.cuda()

                # split and optimise
                model_split, tree_modules_split, node_l, node_r \
                    = optimize_fixed_tree(model_split, tree_struct,
                                          train_loader, valid_loader, test_loader,
                                          args.epochs_node,
                                          node_idx)

                best_tr_loss_after_split = records['train_best_loss']
                best_va_loss_adter_split = records['valid_best_loss_nodes_split'][node_idx]
                best_te_loss_after_split = records['test_best_loss']
                tree_struct[node_idx]['train_accuracy_gain_split'] \
                    = best_tr_loss - best_tr_loss_after_split
                tree_struct[node_idx]['valid_accuracy_gain_split'] \
                    = best_va_loss - best_va_loss_adter_split
                tree_struct[node_idx]['test_accuracy_gain_split'] \
                    = best_te_loss - best_te_loss_after_split

                print("\n----------- Optimizing an extension --------------")
                if not(meta_e['identity']):
                    if args.cuda:
                        model_ext.cuda()

                    # make deeper and optimise
                    model_ext, tree_modules_ext, node_e \
                        = optimize_fixed_tree(model_ext, tree_struct,
                                              train_loader, valid_loader, test_loader,
                                              args.epochs_node,
                                              node_idx)

                    best_tr_loss_after_ext = records['train_best_loss']
                    best_va_loss_adter_ext = records['valid_best_loss_nodes_ext'][node_idx]
                    best_te_loss_after_ext = records['test_best_loss']

                    # TODO: record the gain from split/extra depth:
                    #  need separately record best losses for split & depth
                    tree_struct[node_idx]['train_accuracy_gain_ext'] \
                        = best_tr_loss - best_tr_loss_after_ext
                    tree_struct[node_idx]['valid_accuracy_gain_ext'] \
                        = best_va_loss - best_va_loss_adter_ext
                    tree_struct[node_idx]['test_accuracy_gain_ext'] \
                        = best_te_loss - best_te_loss_after_ext
                else:
                    print('No extension as '
                          'the transformer is an identity function.')
                
                # ---------- Decide whether to split, extend or keep -----------
                criteria = get_decision(args.criteria, node_idx, tree_struct)

                if criteria == 'split':
                    print("\nSplitting node " + str(node_idx))
                    # update the parent node
                    tree_struct[node_idx]['is_leaf'] = False
                    tree_struct[node_idx]['left_child'] = nextind
                    tree_struct[node_idx]['right_child'] = nextind+1
                    tree_struct[node_idx]['split'] = True

                    # add the children nodes
                    tree_struct.append(meta_l)
                    tree_modules_split.append(node_l)
                    tree_struct.append(meta_r)
                    tree_modules_split.append(node_r)

                    # update tree_modules:
                    tree_modules = tree_modules_split
                    nextind += 2
                    change = True
                elif criteria == 'extend':
                    print("\nExtending node " + str(node_idx))
                    # update the parent node
                    tree_struct[node_idx]['is_leaf'] = False
                    tree_struct[node_idx]['left_child'] = nextind
                    tree_struct[node_idx]['extended'] = True

                    # add the children nodes
                    tree_struct.append(meta_e)
                    tree_modules_ext.append(node_e)

                    # update tree_modules:
                    tree_modules = tree_modules_ext
                    nextind += 1
                    change = True
                else:
                    # revert weights back to state before split
                    print("No splitting at node " + str(node_idx))
                    print("Revert the weights to the pre-split state.")
                    model = _load_checkpoint('model.pth')
                    tree_modules = model.update_tree_modules()

                # record the visit to the node
                tree_struct[node_idx]['visited'] = True

                # save the model and tree structures:
                checkpoint_model('model.pth', struct=tree_struct, modules=tree_modules,
                                 data_loader=test_loader,
                                 figname='hist_split_node_{:03d}.png'.format(node_idx))
                checkpoint_msc(tree_struct, records)
                last_node = node_idx

                # global refinement prior to the next growth
                # NOTE: this is an option not included in the paper.
                if args.finetune_during_growth and (criteria == 1 or criteria == 2):
                    print("\n-------------- Global refinement --------------")   
                    model = Tree(tree_struct, tree_modules,
                                 split=False, node_split=last_node,
                                 extend=False, node_extend=last_node,
                                 cuda_on=args.cuda)
                    if args.cuda: 
                        model.cuda()

                    model, tree_modules = optimize_fixed_tree(
                        model, tree_struct,
                        train_loader, valid_loader, test_loader,
                        args.epochs_finetune_node, node_idx,
                    )
        # terminate the tree growth if no split or extend in the final layer
        if not change: break

    # ############### 2: Refinement (finetuning) phase starts #################
    print("\n\n------------------- Fine-tuning the tree --------------------")
    best_valid_accuracy_before = records['valid_best_accuracy']
    model = Tree(tree_struct, tree_modules,
                 split=False,
                 node_split=last_node,
                 child_left=None, child_right=None,
                 extend=False,
                 node_extend=last_node, child_extension=None,
                 cuda_on=args.cuda)
    if args.cuda: 
        model.cuda()

    model, tree_modules = optimize_fixed_tree(model, tree_struct,
                                              train_loader, valid_loader, test_loader,
                                              args.epochs_finetune,
                                              last_node)

    best_valid_accuracy_after = records['valid_best_accuracy']

    # only save if fine-tuning improves validation accuracy
    if best_valid_accuracy_after - best_valid_accuracy_before > 0:
        checkpoint_model('model.pth', struct=tree_struct, modules=tree_modules,
                         data_loader=test_loader,
                         figname='hist_split_node_finetune.png')
    checkpoint_msc(tree_struct, records)


# --------------------------- Start growing an ANT! ---------------------------
start = time.time()
grow_ant_nodewise()
