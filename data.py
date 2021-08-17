import os

import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils import data
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler

class imagenet_data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).cuda().view(1,3,1,1)
        self.std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).cuda().view(1,3,1,1)
        # With Amp, it isn't necessary to manually convert data to half.
        # if args.fp16:
        #     self.mean = self.mean.half()
        #     self.std = self.std.half()
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        # if record_stream() doesn't work, another option is to make sure device inputs are created
        # on the main stream.
        # self.next_input_gpu = torch.empty_like(self.next_input, device='cuda')
        # self.next_target_gpu = torch.empty_like(self.next_target, device='cuda')
        # Need to make sure the memory allocated for next_* is not still in use by the main stream
        # at the time we start copying to next_*:
        # self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)
            # more code for the alternative if record_stream() doesn't work:
            # copy_ will record the use of the pinned source tensor in this side stream.
            # self.next_input_gpu.copy_(self.next_input, non_blocking=True)
            # self.next_target_gpu.copy_(self.next_target, non_blocking=True)
            # self.next_input = self.next_input_gpu
            # self.next_target = self.next_target_gpu

            # With Amp, it isn't necessary to manually convert data to half.
            # if args.fp16:
            #     self.next_input = self.next_input.half()
            # else:
            self.next_input = self.next_input.float()
            self.next_input = self.next_input.sub_(self.mean).div_(self.std)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        if input is not None:
            input.record_stream(torch.cuda.current_stream())
        if target is not None:
            target.record_stream(torch.cuda.current_stream())
        self.preload()
        return input, target

def fast_collate(batch, memory_format):

    imgs = [img[0] for img in batch]
    targets = torch.tensor([target[1] for target in batch], dtype=torch.int64)
    w = imgs[0].size[0]
    h = imgs[0].size[1]
    tensor = torch.zeros( (len(imgs), 3, h, w), dtype=torch.uint8).contiguous(memory_format=memory_format)
    for i, img in enumerate(imgs):
        nump_array = np.asarray(img, dtype=np.uint8)
        if(nump_array.ndim < 3):
            nump_array = np.expand_dims(nump_array, axis=-1)
        nump_array = np.rollaxis(nump_array, 2)
        tensor[i] += torch.from_numpy(nump_array)
    return tensor, targets

def get_imagenet_loaders(specs):
    # Data loading code
    memory_format = torch.contiguous_format
    crop_size = 224 # size of the training images
    val_size = 256 # size of the validation imagaes
    data_dir = specs.get('data_dir', './') # need to put the location of the data here
    traindir = os.path.join(data_dir, 'ILSVRC2012_img_train')
    valdir = os.path.join(data_dir, 'ILSVRC2012_val_reorganize')
    train_dataset = torchvision.datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(crop_size),
            transforms.RandomHorizontalFlip(),
            # transforms.ToTensor(), Too slow
            # normalize,
        ]))
    val_dataset = torchvision.datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(val_size),
            transforms.CenterCrop(crop_size),
        ]))
    collate_fn = lambda b: fast_collate(b, memory_format)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=specs.get('batch_size', 512), shuffle=True,
        num_workers=specs.get('dl_workers', 8), pin_memory=True, sampler=None, collate_fn=collate_fn)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=specs.get('batch_size', 512), shuffle=False,
        num_workers=specs.get('dl_workers', 8), pin_memory=True, sampler=None,
        collate_fn=collate_fn)
    return train_loader, val_loader

def get_cifar10_loaders(valid=False):
    means = (0.4914, 0.4822, 0.4465)
    stds = (0.2023, 0.1994, 0.2010)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(means, stds),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(means, stds),
    ])

    if not valid:
        # create data loaders for train and validation
        trainset = torchvision.datasets.CIFAR10(
                root='./data', train=True, download=True, transform=transform_train)
        trn_dl = torch.utils.data.DataLoader(
                trainset, batch_size=128, shuffle=True, num_workers=1)
        testset = torchvision.datasets.CIFAR10(
                root='./data', train=False, download=True, transform=transform_test)
        test_dl = torch.utils.data.DataLoader(
                testset, batch_size=100, shuffle=False, num_workers=1)
        return trn_dl, test_dl
    else:
        # load the dataset
        valid_size = 0.1
        trainset = torchvision.datasets.CIFAR10(
                root='./data', train=True, download=True, transform=transform_train)
        validset = torchvision.datasets.CIFAR10(
                root='./data', train=True, download=True, transform=transform_test)
        num_train = len(trainset)
        indices = list(range(num_train))
        split = int(np.floor(valid_size * num_train))
        np.random.seed(1)
        np.random.shuffle(indices)
        train_idx, valid_idx = indices[split:], indices[:split] 
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)
        trn_dl = torch.utils.data.DataLoader(
            trainset, batch_size=128, sampler=train_sampler,
            num_workers=1)
        val_dl = torch.utils.data.DataLoader(
            validset, batch_size=100, sampler=valid_sampler,
            num_workers=1)
        return trn_dl, val_dl

def get_cifar100_loaders(valid=False):
    means = [0.507, 0.487, 0.441]
    stds = [0.267, 0.256, 0.276]
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(means, stds),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(means, stds),
    ])

    if not valid:
        # create data loaders for train and validation
        trainset = torchvision.datasets.CIFAR100(
                root='./data', train=True, download=True, transform=transform_train)
        trn_dl = torch.utils.data.DataLoader(
                trainset, batch_size=128, shuffle=True, num_workers=1)
        testset = torchvision.datasets.CIFAR100(
                root='./data', train=False, download=True, transform=transform_test)
        test_dl = torch.utils.data.DataLoader(
                testset, batch_size=100, shuffle=False, num_workers=1)
        return trn_dl, test_dl
    else:
        # load the dataset
        valid_size = 0.1
        trainset = torchvision.datasets.CIFAR100(
                root='./data', train=True, download=True, transform=transform_train)
        validset = torchvision.datasets.CIFAR100(
                root='./data', train=True, download=True, transform=transform_test)
        num_train = len(trainset)
        indices = list(range(num_train))
        split = int(np.floor(valid_size * num_train))
        np.random.seed(1)
        np.random.shuffle(indices)
        train_idx, valid_idx = indices[split:], indices[:split] 
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)
        trn_dl = torch.utils.data.DataLoader(
            trainset, batch_size=128, sampler=train_sampler,
            num_workers=1)
        val_dl = torch.utils.data.DataLoader(
            validset, batch_size=100, sampler=valid_sampler,
            num_workers=1)
        return trn_dl, val_dl

def get_svhn_loaders():
    means = (0.4376821, 0.4437697, 0.47280442)
    stds = (0.19803012, 0.20101562, 0.19703614)
    transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(means, stds),
    ])
    transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(means, stds),
    ])
    trainset = torchvision.datasets.SVHN(
            root='./data/',
            split="train",
            download=True,
            transform=transform_train,
    )
    validset = torchvision.datasets.SVHN(
            root='./data/',
            split="test",
            download=True,
            transform=transform_train,
    )
    trn_dl = torch.utils.data.DataLoader(
            trainset, batch_size=128, shuffle=True,
            num_workers=1)
    val_dl = torch.utils.data.DataLoader(
            validset, batch_size=100, shuffle=False,
            num_workers=1)
    return trn_dl, val_dl

def get_tin_data():
    fp = './data/'
    means = (0.4810, 0.4482, 0.3968)
    stds = (0.2765,  0.2684, 0.2817)
    transform_train = transforms.Compose([
        transforms.RandomCrop(64, padding=8, padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(means, stds),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(means, stds),
    ])
    # only load images with the desired image size (some have no channel)
    image_size = (64, 64, 3)

    # get file name to class map
    with open(fp + 'tiny-imagenet-200/wnids.txt') as f:
        dir_names = f.readlines()
        name_to_idx_map = {dn.strip().lower(): i for i, dn in enumerate(dir_names)}

    # get the training data
    print('Loading the training data ...')
    all_trn_labels = []
    all_trn_images = []
    for dn, label_idx in tqdm(name_to_idx_map.items()):
        class_fp = fp + f'tiny-imagenet-200/train/{dn}/images/'
        fns = os.listdir(class_fp)
        for fn in fns:
            full_path = class_fp + fn
            img = Image.open(full_path, 'r')
            if np.asarray(img).shape == image_size:
                all_trn_images.append(img)
                all_trn_labels.append(label_idx)
            img.load()

    # get labels and file names for all the validation data
    fn_to_label_map = {}
    with open(fp + 'tiny-imagenet-200/val/val_annotations.txt', 'r') as f:
        all_labels = f.readlines()
        for label in all_labels:
            split_label = label.split()
            img_name = split_label[0]
            class_idx = name_to_idx_map[split_label[1]]
            fn_to_label_map[img_name] = class_idx

    # load all of the validation data
    print('Loading the validation data ...')
    all_valid_labels = []
    all_valid_images = []
    for valid_name, label_idx in tqdm(fn_to_label_map.items()):
        full_path = fp + f'tiny-imagenet-200/val/images/{valid_name}'
        img = Image.open(full_path, 'r')
        if np.asarray(img).shape == image_size:
            all_valid_images.append(img)
            all_valid_labels.append(label_idx)
        img.load()

    # create the dataloaders and datasets
    trn_ds = TINDataset(all_trn_images, all_trn_labels, transform_train)
    val_ds = TINDataset(all_valid_images, all_valid_labels, transform_test)
    trn_dl = data.DataLoader(trn_ds, shuffle=True, batch_size=64)
    val_dl = data.DataLoader(val_ds, shuffle=False, batch_size=64)
    return trn_dl, val_dl

class TINDataset(data.Dataset):
    def __init__(self, imgs, labels, tfms):
        super().__init__()
        self.imgs = imgs
        self.labels = labels
        self.tfms = tfms

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = self.tfms(self.imgs[idx])
        label = self.labels[idx]
        return img, label

if __name__=='__main__':
    trn_dl, test_dl = get_svhn_loaders()
    print(len(trn_dl))
    print(len(test_dl))
