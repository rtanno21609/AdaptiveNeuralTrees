"""Data loader"""
import torch
import torchvision
from torchvision import datasets, transforms
from ops import ChunkSampler


def get_dataloaders(
        dataset='mnist',
        batch_size=128,
        augmentation_on=False,
        cuda=False, num_workers=0,
):
    # TODO: move the dataloader to data.py
    kwargs = {
        'num_workers': num_workers, 'pin_memory': True,
    } if cuda else {}

    if dataset == 'mnist':
        if augmentation_on:
            transform_train = transforms.Compose(
                [
                    transforms.RandomCrop(28, padding=2),
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,)),
                ],
            )
            transform_test = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,)),
                ],
            )
        else:
            transform_train = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,)),
                ],
            )
            transform_test = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,)),
                ],
            )

        mnist_train = datasets.MNIST(
            '../data', train=True, download=True, transform=transform_train,
        )
        mnist_valid = datasets.MNIST(
            '../data', train=True, download=True, transform=transform_test,
        )
        mnist_test = datasets.MNIST(
            '../data', train=False, transform=transform_test,
        )

        TOTAL_NUM = 60000
        NUM_VALID = int(round(TOTAL_NUM * 0.1))
        NUM_TRAIN = int(round(TOTAL_NUM - NUM_VALID))

        train_loader = torch.utils.data.DataLoader(
            mnist_train,
            batch_size=batch_size,
            sampler=ChunkSampler(NUM_TRAIN, 0, shuffle=True),
            **kwargs)

        valid_loader = torch.utils.data.DataLoader(
            mnist_valid,
            batch_size=batch_size,
            sampler=ChunkSampler(NUM_VALID, NUM_TRAIN, shuffle=True),
            **kwargs)

        test_loader = torch.utils.data.DataLoader(
            mnist_test,
            batch_size=1000,
            shuffle=False,
            **kwargs)

    elif dataset == 'cifar10':
        if augmentation_on:
            transform_train = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010),
                    ),
                ],
            )
            transform_test = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010),
                    ),
                ],
            )
        else:
            transform_train = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ],
            )
            transform_test = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ],
            )

        cifar10_train = torchvision.datasets.CIFAR10(
            root='../data', train=True, download=True,
            transform=transform_train,
        )
        cifar10_valid = torchvision.datasets.CIFAR10(
            root='../data', train=True, download=True, transform=transform_test,
        )
        cifar10_test = torchvision.datasets.CIFAR10(
            root='../data', train=False, download=True,
            transform=transform_test,
        )

        TOTAL_NUM = 50000
        NUM_VALID = int(round(TOTAL_NUM * 0.02))
        NUM_TRAIN = int(round(TOTAL_NUM - NUM_VALID))
        train_loader = torch.utils.data.DataLoader(
            cifar10_train,
            batch_size=batch_size,
            sampler=ChunkSampler(NUM_TRAIN, 0, shuffle=True),
            **kwargs)
        valid_loader = torch.utils.data.DataLoader(
            cifar10_valid,
            batch_size=batch_size,
            sampler=ChunkSampler(NUM_VALID, NUM_TRAIN, shuffle=True),
            **kwargs)
        test_loader = torch.utils.data.DataLoader(
            cifar10_test,
            batch_size=1000,
            shuffle=False,
            **kwargs)
    else:
        raise NotImplementedError("Specified data set is not available.")

    return train_loader, valid_loader, test_loader, NUM_TRAIN, NUM_VALID


def get_dataset_details(dataset):
    if dataset == 'mnist':
        input_nc, input_width, input_height = 1, 28, 28
        classes = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
    elif dataset == 'cifar10':
        input_nc, input_width, input_height = 3, 32, 32
        classes = (
            'plane', 'car', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck',
        )
    else:
        raise NotImplementedError("Specified data set is not available.")

    return input_nc, input_width, input_height, classes
