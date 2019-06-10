"""Utility functions"""
import torch
import torchvision
import json
import time
import models
import random
import numpy as np
import matplotlib.pyplot as plt

from torch.autograd import Variable
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from ops import \
    count_number_transforms, count_number_transforms_after_last_downsample


# ---------------------------- Model configuration ----------------------
def define_node(
        args, node_index, level, parent_index, tree_struct, identity=False,
):
    """ Define node operations.
    
    In this function, we assume that 3 building blocks of node operations
    i.e. transformer, solver and router are of fixed complexity. 
    """

    # define meta information
    num_transforms = 0 if node_index == 0 else count_number_transforms(parent_index, tree_struct)
    meta = {'index': node_index,
            'parent': parent_index,
            'left_child': 0,
            'right_child': 0,
            'level': level,
            'extended': False,
            'split': False,
            'visited': False,
            'is_leaf': True,
            'train_accuracy_gain_split': -np.inf,
            'valid_accuracy_gain_split': -np.inf,
            'test_accuracy_gain_split': -np.inf,
            'train_accuracy_gain_ext': -np.inf,
            'valid_accuracy_gain_ext': -np.inf,
            'test_accuracy_gain_ext': -np.inf,
            'num_transforms': num_transforms}

    # get input shape before transformation
    if not tree_struct: # if it's first node, then set it to the input data size
        meta['in_shape'] = (1, args.input_nc, args.input_width, args.input_height)
    else:
        meta['in_shape'] = tree_struct[parent_index]['out_shape']

    # -------------------------- define transformer ---------------------------
    # no transformation if the input size is too small. 
    if meta['in_shape'][2] < 3 or meta['in_shape'][3] < 3:
        identity = True

    if identity or args.transformer_ver==1:
        meta['transformed'] = False
    else:
        meta['transformed'] = True

    # only downsample at the specified frequency:
    # currently assume the initial transform always perform downsampling.
    num_downsample = 0 if node_index == 0 else count_number_transforms_after_last_downsample(parent_index, tree_struct)
    if args.downsample_interval == num_downsample or node_index == 0:
        meta['downsampled'] = True
    else:
        meta['downsampled'] = False

    # get the transformer version: 
    config_t = {'kernel_size': args.transformer_k,
                'ngf': args.transformer_ngf,
                'batch_norm': args.batch_norm,
                'downsample': meta['downsampled'],
                'expansion_rate': args.transformer_expansion_rate,
                'reduction_rate': args.transformer_reduction_rate
                }
    transformer_ver = args.transformer_ver
    if identity:
        transformer = models.Identity(meta['in_shape'][1], meta['in_shape'][2], meta['in_shape'][3],
                                      **config_t)
    else:
        transformer = define_transformer(transformer_ver,
                                         meta['in_shape'][1], meta['in_shape'][2], meta['in_shape'][3],
                                         **config_t)
    meta['identity'] = identity
    
    # get output shape after transformation:
    meta['out_shape'] = transformer.outputshape
    print('---------------- data shape before/after transformer -------------')
    print(meta['in_shape'], type(meta['in_shape']))
    print(meta['out_shape'], type(meta['out_shape']))

    # ---------------------------- define solver-------------------------------
    config_s = {'no_classes': args.no_classes,
                'dropout_prob': args.solver_dropout_prob,
                'batch_norm': args.batch_norm}
    solver = define_solver(args.solver_ver, 
                           meta['out_shape'][1], meta['out_shape'][2], meta['out_shape'][3],
                           **config_s)

    # ---------------------------- define router ------------------------------
    config_r = {'kernel_size': args.router_k, 
                'ngf': args.router_ngf,
                'soft_decision': True,
                'stochastic': False,
                'dropout_prob':args.router_dropout_prob,
                'batch_norm': args.batch_norm}
   
    router = define_router(
        args.router_ver,
        meta['out_shape'][1], meta['out_shape'][2], meta['out_shape'][3],
        **config_r)

    # define module: 
    module = {'transform': transformer,
              'classifier': solver,
              'router': router}

    return meta, module


def define_transformer(version, input_nc, input_width, input_height, **kwargs):
    if version == 1:  # Identity function
        return models.Identity(input_nc, input_width, input_height, **kwargs)
    elif version == 2:  # 1 conv layer
        return models.JustConv(input_nc, input_width, input_height, **kwargs)
    elif version == 3:  # 1 conv layer + 1 max pooling
        return models.ConvPool(input_nc, input_width, input_height, **kwargs)
    elif version == 4:  # Bottle-neck residual block
        return models.ResidualTransformer(input_nc, input_width, input_height, **kwargs)
    elif version == 5:  # VGG13: 2 conv layer + 1 max pooling
        return models.VGG13ConvPool(input_nc, input_width, input_height, **kwargs)
    else:
        raise NotImplementedError("Specified transformer module not available.")


def define_router(version, input_nc, input_width, input_height, **kwargs):
    if version == 1:  # Simple router with 1 conv kernel + spatial averaging
        return models.Router(input_nc, input_width, input_height, **kwargs)
    elif version == 2:  # 1 conv layer with global average pooling + fc layer
        return models.RouterGAP(input_nc, input_width, input_height, **kwargs)
    elif version == 3:  # 2 conv with global average pooling + fc layer
        return models.RouterGAPwithDoubleConv(input_nc, input_width, input_height, **kwargs)
    elif version == 4:  # MLP with 1 hidden layer
        return models.Router_MLP_h1(input_nc, input_width, input_height, **kwargs)
    elif version == 5:  # GAP + 2 fc layers (Veit. et al 2017)
        return models.RouterGAP_TwoFClayers(input_nc, input_width, input_height, **kwargs)
    elif version == 6:  # 1 conv + GAP + 2 fc layers
        return models.RouterGAPwithConv_TwoFClayers(input_nc, input_width, input_height, **kwargs)
    else:
        raise NotImplementedError("Specified router module not available!")


def define_solver(version, input_nc, input_width, input_height, **kwargs):
    if version == 1:  # Logistric regressor
        return models.LR(input_nc, input_width, input_height, **kwargs)
    elif version == 2:  # MLP with 2 hidden layers:
        return models.MLP_LeNet(input_nc, input_width, input_height, **kwargs)
    elif version == 3:  # MLP with a single hidden layer (MNIST LeNet)
        return models.MLP_LeNetMNIST(input_nc, input_width, input_height, **kwargs)
    elif version == 4:  # GAP + 2 FC layers
        return models.Solver_GAP_TwoFClayers(input_nc, input_width, input_height, **kwargs)
    elif version == 5:  # MLP with a single hidden layer in AlexNet
        return models.MLP_AlexNet(input_nc, input_width, input_height, **kwargs)
    elif version == 6:  # GAP + 1 FC layer
        return models.Solver_GAP_OneFClayers(input_nc, input_width, input_height, **kwargs)
    else:
        raise NotImplementedError("Specified solver module not available!")


def get_scheduler(scheduler_type, optimizer, grow):
    if scheduler_type == 'step_lr': # reduce the learning rate
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[100,150], gamma=0.1,
        )
    elif scheduler_type == 'plateau': # patience based decay of learning rates
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', factor=0.1, patience=10,
        )
    elif scheduler_type == 'hybrid': # hybrid between step_lr and plateau
        if grow: # use 'plateau' during the local growth phase
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 'min', factor=0.1, patience=10,
            )
        else:
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=[100, 150], gamma=0.1,
            )
    else:
        scheduler = None
    return scheduler


# --------------------------- Visualisation ----------------------------
# visualise numpy image:
def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


def plot_hist(data, save_as='./figure'):
    fig = plt.figure()
    plt.hist(data, normed=True, bins=150, range=(0, 1.0))
    fig.savefig(save_as)


def plot_hist_root(labels, split_status, save_as='./figures/hist_labels_split.png'):
    """ Plot the distribution of labels of a binary routing function.
    Args:
        labels (np array): labels (N) each entry contains a label
        split_status (np array bool): boolean array (N) where 0 indicates the entry
        belongs to the right and 1 indicates left.
    """
    fig = plt.figure()
    plt.hist(labels[split_status], bins=range(11), alpha=0.75, label='right branch')
    plt.hist(labels[split_status==False], bins=range(11), alpha=0.5, label='left branch')
    plt.legend(loc='upper right')
    print('save the histogram as ' + save_as)
    fig.savefig(save_as)


# Use to visualise performance of one model:
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
    

def plot_performance(jasonfiles, model_names=[], figsize=(5,5), title='') :
    """ Visualise the results for several models

    Args:
        jasonfiles (list): List of jason files
        model_names (list): List of model names
    """
    # TODO: currently only supports up to 8 models at a time due to color types
    fig = plt.figure(figsize=figsize)
    color = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']

    if not(model_names):
        model_names = [str(i) for i in range(len(jasonfiles))]

    for i, f in enumerate(jasonfiles):
        # load the information: 
        records = json.load(open(f, 'r'))
        
        # Plot train/test loss
        plt.plot(np.arange(len(records['test_epoch_loss'])), np.array(records['test_epoch_loss']),
                 color=color[i], linestyle='-.', label='test epoch loss: ' + model_names[i] )     
        plt.plot(np.arange(len(records['train_epoch_loss']), dtype=float), np.array(records['train_epoch_loss']), 
                 color=color[i], linestyle='-', label='train epoch loss: '  + model_names[i])
    plt.ylabel('epoch wise loss (average CE loss)')
    plt.xlabel('epoch number')
    plt.legend(loc='upper right')
    plt.title(title)


def plot_accuracy(jasonfiles, model_names=[], figsize=(5,5), ymax=100.0, title=''):
    fig = plt.figure(figsize=figsize)
    color = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']

    if not(model_names):
        model_names = [str(i) for i in range(len(jasonfiles))]

    for i, f in enumerate(jasonfiles):
        # load the information: 
        records = json.load(open(f, 'r'))
        
        # Plot train/test loss
        plt.plot(
            np.arange(len(records['test_epoch_accuracy']), dtype=float),
            np.array(records['test_epoch_accuracy']),
            color=color[i], linestyle='-', label=model_names[i],
        )
        # print(records['train_epoch_accuracy'])
        plt.ylabel('test accuracy (%)')
        plt.xlabel('epoch number')
        plt.ylim(ymax=ymax)
        print(model_names[i] + ': accuracy = {}'.format(max(records['test_epoch_accuracy'])))
    plt.legend(loc='lower right')
    plt.title(title)


def compute_error(model_file, data_loader, cuda_on=False, name = ''):
    """Load a model and compute errors on a held-out dataset
    Args:
        model_file (str): model parameters
        data_dataloader (torch.utils.data.DataLoader): data loader
    """
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


def load_tree_model(model_file, cuda_on=False,
                    soft_decision=True, stochastic=False,
                    breadth_first=False, fast=False,
                    ):
    """Load a tree model. """
    # load the model and set routers stochastic.
    map_location = None
    if not (cuda_on):
        map_location = 'cpu'

    tree_tmp = torch.load(model_file, map_location=map_location)
    tree_struct, tree_modules = tree_tmp.tree_struct, tree_tmp.update_tree_modules()
    for node in tree_modules:
        node['router'].stochastic = stochastic
        node['router'].soft_decision = soft_decision
        node['router'].dropout_prob = 0.0

    for node_meta in tree_struct:
        if not ('extended' in node_meta.keys()):
            node_meta['extended'] = False

    model = models.Tree(
        tree_struct, tree_modules,
        split=False, cuda_on=cuda_on, soft_decision=soft_decision,
        breadth_first=breadth_first,
    )
    if cuda_on:
        model.cuda()
    return model


def compute_error_general(model_file, data_loader, cuda_on=False,
                          soft_decision=True, stochastic=False,
                          breadth_first=False, fast = False,
                          task="classification",
                          name = ''):
    """Load a model and perform stochastic inferenc
    Args:
        model_file (str): model parameters
        data_dataloader (torch.utils.data.DataLoader): data loader

    """
    # load the model and set routers stochastic.
    map_location = None
    if not (cuda_on):
        map_location = 'cpu'

    tree_tmp = torch.load(model_file, map_location=map_location)
    tree_struct, tree_modules = \
        tree_tmp.tree_struct, tree_tmp.update_tree_modules()

    for node in tree_modules:
        node['router'].stochastic = stochastic
        node['router'].soft_decision = soft_decision
        node['router'].dropout_prob = 0.0

    for node_meta in tree_struct:
        if not('extended' in node_meta.keys()):
            node_meta['extended']=False

    if task == "classification":
        model = models.Tree(
            tree_struct, tree_modules,
            split=False, cuda_on=cuda_on, soft_decision=soft_decision,
            breadth_first=breadth_first,
        )

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
        
        if fast:
            output = model.fast_forward_BF(data)
        else:
            output = model.forward(data)

        if task == "classification":
            test_loss += F.nll_loss(output, target, size_average=False).data[0]
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        else:
            raise NotImplementedError("The specified task is not supported")

    # Normalise the loss and print:
    if task == "classification":
        test_loss /= len(data_loader.dataset)
        print(name + 'Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
            test_loss, correct, len(data_loader.dataset),
            100. * correct / len(data_loader.dataset)))
    elif task == "regression":
        test_loss = test_loss / 7.0 / len(data_loader.dataset)
        print('Average loss: {:.4f}'.format(test_loss))


def compute_error_general_ensemble(model_file_list, data_loader, cuda_on=False,
                                   soft_decision=True, stochastic=False,
                                   breadth_first=False, fast = False,
                                   task="classification",
                                   name = ''):
    """Load an ensemble of models and compute the average prediction. """

    # load the model and set routers stochastic.
    model_list = []
    map_location = None
    if not (cuda_on):
        map_location = 'cpu'

    for model_file in model_file_list:
        tree_tmp = torch.load(model_file, map_location=map_location)
        tree_struct, tree_modules = tree_tmp.tree_struct, tree_tmp.update_tree_modules()
        for node in tree_modules:
            node['router'].stochastic = stochastic
            node['router'].soft_decision = soft_decision
            node['router'].dropout_prob = 0.0

        for node_meta in tree_struct:
            if not('extended' in node_meta.keys()):
                node_meta['extended']=False

        if task == "classification":
            model = models.Tree(
                tree_struct, tree_modules,
                split=False, cuda_on=cuda_on, soft_decision=soft_decision,
                breadth_first=breadth_first,
            )

        if cuda_on:
            model.cuda()

        model_list.append(model)

    # compute the error
    for model in model_list:
        model.eval()

    test_loss = 0
    correct = 0

    for data, target in data_loader:
        if cuda_on:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)

        # compute the average prediction over different models
        output = 0.0
        for model in model_list:
            if fast:
                output += model.fast_forward_BF(data)
            else:
                output += model.forward(data)
        output /= len(model_list)

        if task == "classification":
            test_loss += F.nll_loss(output, target, size_average=False).data[0]
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        elif task == "regression":
            # print(test_loss)
            test_loss += F.mse_loss(output, target, size_average=False).data[0]

    # Normalise the loss and print:
    if task == "classification":
        test_loss /= len(data_loader.dataset)
        print(name + 'Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
            test_loss, correct, len(data_loader.dataset),
            100. * correct / len(data_loader.dataset)))
    elif task == "regression":
        test_loss = test_loss / 7.0 / len(data_loader.dataset)
        print('Average loss: {:.4f}'.format(test_loss))


def try_different_inference_methods(
        model_file, dataset,  task="classification",
        augmentation_on=False, cuda_on=True,
):
    """ Try different inference methods and compute accuracy 
    """
    if dataset == 'cifar10':
        if augmentation_on:
            transform_test = transforms.Compose([transforms.ToTensor(),
                                                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        else:
            transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        cifar10_test = torchvision.datasets.CIFAR10(root='../../data', train=False, download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(cifar10_test, batch_size=100, shuffle=False, num_workers = 2)
    elif dataset == 'mnist':
        transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        mnist_test = datasets.MNIST('../../data', train=False, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=100, shuffle=False, num_workers=2)
    else:
        raise NotImplementedError("The specified dataset is not supported")

    # soft inferences:
    start = time.time()
    compute_error_general(
        model_file, test_loader, task=task,
        cuda_on=cuda_on, soft_decision=True, stochastic=False,
        breadth_first=True, name='soft + BF : ',
    )

    end = time.time()
    print('took {} seconds'.format(end - start))

    # hard:
    compute_error_general(
        model_file, test_loader, task=task,
        cuda_on=cuda_on, soft_decision=False, stochastic=False,
        breadth_first=True,
        name='hard + max + BF : ',
    )
    end = time.time()
    print('took {} seconds'.format(end - start))

    # stochastic hard
    compute_error_general(
        model_file, test_loader,
        cuda_on=cuda_on, soft_decision=False, stochastic=True,
        breadth_first=True, name='hard + stochastic + BF : ',
    )
    end = time.time()
    print('took {} seconds'.format(end - start))


def try_different_inference_methods_ensemble(
        model_file_list, dataset, task="classification",
        augmentation_on=False, cuda_on=True,
):
    """ Try different inference methods and compute accuracy
    """

    if dataset == 'cifar10':
        if augmentation_on:
            transform_test = transforms.Compose([transforms.ToTensor(),
                                                 transforms.Normalize(
                                                     (0.4914, 0.4822, 0.4465),
                                                     (0.2023, 0.1994, 0.2010))])
        else:
            transform_test = transforms.Compose([transforms.ToTensor(),
                                                 transforms.Normalize(
                                                     (0.5, 0.5, 0.5),
                                                     (0.5, 0.5, 0.5))])

        cifar10_test = torchvision.datasets.CIFAR10(root='../../data',
                                                    train=False, download=True,
                                                    transform=transform_test)
        test_loader = torch.utils.data.DataLoader(cifar10_test, batch_size=100,
                                                  shuffle=False, num_workers=2)
    elif dataset == 'mnist':
        transform_test = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        mnist_test = datasets.MNIST('../../data', train=False,
                                    transform=transform_test)
        test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=100,
                                                  shuffle=False, num_workers=2)
    else:
        raise NotImplementedError("The specified dataset is not availble")

    # soft inferences:
    start = time.time()
    compute_error_general_ensemble(
        model_file_list, test_loader, task=task,
        cuda_on=cuda_on, soft_decision=True, stochastic=False,
        breadth_first=True, name='soft + BF : ',
    )

    end = time.time()
    print('took {} seconds'.format(end - start))

    # hard:
    compute_error_general_ensemble(
        model_file_list, test_loader, task=task,
        cuda_on=cuda_on, soft_decision=False, stochastic=False,
        breadth_first=True,
        name='hard + max + BF : ',
    )
    end = time.time()
    print('took {} seconds'.format(end - start))


# --------------------- Parameter counters  -------------------------
def get_total_number_of_params(model, print_on=False):
    tree_struct = model.tree_struct
    
    names, params = [], []
    for node_idx, node_meta in enumerate(tree_struct):
        for name, param in model.named_parameters():
            if (( not(node_meta['is_leaf']) and '.'+str(node_idx)+'.router' in name) \
            or ('.'+str(node_idx)+'.transform' in name) \
            or (node_meta['is_leaf'] and '.'+str(node_idx)+'.classifier' in name)):
                names.append(name)
                params.append(param)
                
    if print_on:
        print("Count the number of parameters below: ")          
        for name in names: print('          '+name)
            
    return sum(p.numel() for p in params)


def get_number_of_params_path(
        model, nodes, print_on=False, include_routers=True,
):
    names, params = [], []
    if include_routers:
        for name, param in model.named_parameters():
            if '.'+str(nodes[-1])+'.classifier' in name \
            or any(['.'+str(node)+'.transform' in name for node in nodes]) \
            or any(['.'+str(node)+'.router' in name for node in nodes[:-1]]):
                names.append(name)
                params.append(param)
    else:
        for name, param in model.named_parameters():
            if '.' + str(nodes[-1]) + '.classifier' in name \
            or any(['.' + str(node) + '.transform' in name for node in nodes]):
                names.append(name)
                params.append(param)

    if print_on:
        print("\nCount the number of parameters below: ")          
        for name in names: print('          '+name)
    
    return sum(p.numel() for p in params)


def get_number_of_params_summary(
        model, name='', print_on=True, include_routers=True,
):
    # compute the total number 
    total_num = get_total_number_of_params(model)
    
    # compute min,max,mean number of parameters per branch
    paths_list = model.paths_list
    num_list = []
    for (nodes, _) in paths_list:
        num = get_number_of_params_path(
            model, nodes, include_routers=include_routers,
        )
        num_list.append(num)
    
    if print_on:
        print('\n' + name)
        print('Number of parameters summary:')
        print('    Total: {} '.format(total_num))
        print('    Max per branch: {} '.format(max(num_list)))
        print('    Min per branch: {} '.format(min(num_list)))
        print('    Average per branch: {}'.format(sum(num_list)*1.0/len(num_list)))
    
    return total_num, max(num_list), min(num_list), sum(num_list)*1.0/len(num_list)


# --------------------- Others -------------------------
def round_value(value, binary=False):
    divisor = 1024. if binary else 1000.

    if value // divisor**4 > 0:
        return str(round(value / divisor**4, 2)) + 'T'
    elif value // divisor**3 > 0:
        return str(round(value / divisor**3, 2)) + 'G'
    elif value // divisor**2 > 0:
        return str(round(value / divisor**2, 2)) + 'M'
    elif value // divisor > 0:
        return str(round(value / divisor, 2)) + 'K'
    return str(value)


def set_random_seed(seed, cuda):
    np.random.seed(seed)  # cpu vars
    torch.manual_seed(seed)  # pytorch cpu vars
    random.seed(seed)  # Python
    if cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # pytorch gpu vars
        torch.backends.cudnn.deterministic = True   # needed
        torch.backends.cudnn.benchmark = False
