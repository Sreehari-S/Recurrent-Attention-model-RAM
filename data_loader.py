import numpy as np
import torch.utils.data as torchutils
import torchvision
from torchvision import transforms


class MyMNIST(torchvision.datasets.MNIST):
    def __init__(self, *args, **kwargs):
        self.num_channels = 1
        self.num_class = 10
        super(MyMNIST, self).__init__(*args, **kwargs)


def get_MNIST_train_val_dataset(data_dir):
    import os
    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)
    return MyMNIST(
        data_dir, train=True, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    ), None


def get_MNIST_test_dataset(data_dir):
    return MyMNIST(
        data_dir, train=False, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    )


def get_train_val_loader(train_dataset, val_dataset, val_split=0.1, random_split=False, batch_size=32, num_workers=4, pin_memory=False):
    """
    Utility function for loading and returning train and valid multi-process iterators over the dataset.

    If using CUDA, num_workers should be set to 1 and pin_memory to True.

    @param train_dataset: A pytorch dataset for training purpose.
    @param val_dataset: A pytorch dataset for validation purpose. If None, generate from train_dataset.
    @param val_split: percentage split of the training set used for validation. If val_dataset is present, this is of no use.
    @param random_split: whether to randomly split train/validation samples. If val_dataset is present, this is of no use.
    @param batch_size: how many samples per batch to load.
    @param the validation set. Should be a float in the range [0, 1].
    @param num_workers: number of subprocesses to use when loading the dataset.
    @param pin_memory: whether to copy tensors into CUDA pinned memory. Set it to True if using GPU.

    @return A tuple containing training/validation sample iterator
    """
    if val_dataset is not None:
        train_loader = torchutils.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory
        )
        val_loader = torchutils.DataLoader(
            val_dataset, batch_size=batch_size,
            num_workers=num_workers, pin_memory=pin_memory
        )
        return (train_loader, val_loader)
    else:
        num_train = len(train_dataset)
        indices = list(range(num_train))
        split = int(np.floor(val_split * num_train))

        if random_split:
            np.random.shuffle(indices)

        train_idx, valid_idx = indices[split:], indices[:split]

        train_sampler = torchutils.sampler.SubsetRandomSampler(train_idx)
        valid_sampler = torchutils.sampler.SubsetRandomSampler(valid_idx)

        train_loader = torchutils.DataLoader(
            train_dataset, batch_size=batch_size, sampler=train_sampler,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        val_loader = torchutils.DataLoader(
            train_dataset, batch_size=batch_size, sampler=valid_sampler,
            num_workers=num_workers, pin_memory=pin_memory,
        )
        return (train_loader, val_loader)


def get_test_loader(test_dataset,
                    batch_size,
                    num_workers=4,
                    pin_memory=False):
    """
    Utility function for loading and returning a multi-process test iterator over the dataset.

    If using CUDA, num_workers should be set to 1 and pin_memory to True.
    @param test_dataset: A pytorch dataset for testing purpose.
    @param batch_size: how many samples per batch to load.
    @param num_workers: number of subprocesses to use when loading the dataset.
    @param pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
    True if using GPU.

    @return A test sample iterator.
    """
    data_loader = torchutils.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    return data_loader
