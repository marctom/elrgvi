import logging
from types import SimpleNamespace
from pathlib import Path
import time
import argparse

import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torchvision import transforms as T

from alg import elrg
from fit import fit, DL

logging.basicConfig(
    format="[%(asctime)s, %(levelname)s] %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
    level=logging.INFO,
)

torch.backends.cudnn.benchmark = True


class LeNet(nn.Module):

    def __init__(self, large, num_classes):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(256, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def get_transform(pad, crop, stats, flip):
    tfm = [
        T.Pad(pad, padding_mode="reflect"),
        T.RandomCrop(crop),
    ]

    multiplier = 4

    if flip:
        tfm += [T.RandomHorizontalFlip(0.5)]
        multiplier *= 2

    base = [T.ToTensor(), T.Normalize(*stats)]

    return (
        T.Compose(base + tfm),
        T.Compose(base),
        multiplier
    )


IMAGENET_STATS = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
CIFAR_STATS = ([0.491, 0.482, 0.447], [0.247, 0.243, 0.261])
MNIST_STATS = ([0.1307], [0.3081])

MNISTS = ['MNIST', 'KMNIST', 'FashionMNIST', 'EMNIST']

TRANSFORMS = \
    {"CIFAR10":
         get_transform(4, 32, CIFAR_STATS, flip=True),
     "CIFAR100":
         get_transform(4, 32, CIFAR_STATS, flip=True),
     "SVHN":
         get_transform(4, 32, IMAGENET_STATS, flip=False),
     "STL10":
         get_transform(12, 96, IMAGENET_STATS, flip=True),

     }

TRANSFORMS.update({name: get_transform(4, 28, MNIST_STATS, flip=False)
                   for name in MNISTS})

NUM_CLASS = {
    "CIFAR10": 10,
    "CIFAR100": 100,
    "SVHN": 10,
    "STL10": 10,
    "MNIST": 10,
    "FashionMNIST": 10,
    "KMNIST": 10,
}


def get_stats(ds):
    if ds in MNISTS:
        return MNIST_STATS
    elif ds in ['CIFAR10', 'CIFAR100']:
        return CIFAR_STATS
    else:
            return IMAGENET_STATS



def get_dl(local_storage_path,
           dataset,
           data_transform,
           transforms,
           train,
           batch_size,
           device):
    
    
    def get_ds_kwargs(dataset, train):
        if dataset in ['SVHN', 'STL10']:
            kwargs = {'split': 'train' if train else 'test'}
        elif dataset == 'EMNIST':
            kwargs = {'split': 'letters', 'train': train}
        else:
            kwargs = {'train': train}
        return kwargs
    
    kwargs = get_ds_kwargs(dataset, train)

    ds = getattr(torchvision.datasets, dataset)(
        local_storage_path,
        download=True,
        transform=transforms,
        **kwargs)


    if not data_transform:

        if dataset in ['CIFAR10', 'CIFAR100']:
            return DL(torch.from_numpy(ds.data).float().transpose(1, 3),
                      torch.tensor(ds.targets),
                      batch_size=batch_size,
                      device=device)

        elif dataset in ['SVHN', 'STL10']:
            return DL(torch.from_numpy(ds.data).float(),
                      torch.tensor(ds.labels),
                      batch_size=batch_size)

        elif dataset in ['KMNIST', 'EMNIST', 'FashionMNIST', 'MNIST']:
            return DL(ds.train_data[:, None].float(),
                      ds.targets,
                      batch_size=batch_size,
                      device=device)

        else:
            raise NotImplementedError()

    return torch.utils.data.DataLoader(
        dataset=ds,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=12 if train else 8,
        shuffle=train,
    )



def train_cnn(config):

    num_class = NUM_CLASS[config.dataset]
    device = torch.device(config.device)
                        

    if config.model == 'LeNet':
        model = LeNet(large=config.dataset not in MNISTS,
                      num_classes=num_class)
    else:
        model = getattr(torchvision.models,
                        config.model)(num_classes=num_class)

    model = nn.Sequential(model,
                          nn.LogSoftmax(dim=-1))

    train_tfm, valid_tfm, multiplier = TRANSFORMS[config.dataset]

    data = SimpleNamespace(
        train_dl=get_dl(local_storage_path="/tmp",
                        dataset=config.dataset,
                        data_transform=config.data_transform,
                        train=True,
                        batch_size=config.batch_size,
                        device=device,
                        transforms=train_tfm),
        valid_dl=get_dl(local_storage_path="/tmp",
                        dataset=config.dataset,
                        data_transform=config.data_transform,
                        train=False,
                        batch_size=config.batch_size // 10,
                        device=device,
                        transforms=valid_tfm)
    )

    model.to(device)
    num_data = len(data.train_dl.dataset) * \
               (multiplier if config.data_transform else 1)
    model = elrg(model, config, num_data)

    results = fit(
        model=model,
        data=data,
        loss_func=F.nll_loss,
        num_updates=config.num_updates,
        keep_curve=False,
        device=device,
        log_steps=100,
    )

    logging.info(
        f"Final test nll {results.loss:.3f} "
        f"error rate {results.er:.2f}"
    )
    
if __name__ == "__main__":
    
    def str2bool(v):
        return v.lower() in ("yes", "true", "t", "1")

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",
                        default="CIFAR10",
                        type=str)
    parser.add_argument("--model",
                        default="resnet18")
    parser.add_argument("--rank",
                        type=int,
                        default=1, )
    parser.add_argument("--learn_diag",
                        type=str2bool,
                        default=False)
    parser.add_argument("--num_updates",
                        type=int,
                        default=60000)
    parser.add_argument("--data_transform",
                        type=str2bool,
                        default=True)
    parser.add_argument("--batch_size",
                        type=int,
                        default=512)
    parser.add_argument("--lr",
                        type=float,
                        default=0.0005)
    parser.add_argument("--device",
                        type=str,
                        default="cuda")
    parser.add_argument("--num_test_samples",
                        type=int,
                        default=1000, )
    parser.add_argument("--test_batch_size",
                        type=int,
                        default=512, )
    parser.add_argument("--scale_prior",
                        help="Scale prior variances by the size of the input to the layer (Radford Neal).",
                        type=str2bool,
                        default=True)
    parser.add_argument("--q_init_logvar",
                        help="Initial log variance of mean-field "
                             "Gaussian variational posterior.",
                        type=float,
                        default=-12, )
    parser.add_argument("--prior_precision",
                        help="Precision of prior over weights.",
                        type=float,
                        default=1.0, )

    args = parser.parse_args()
    logging.info(f"Config: {args}")
    train_cnn(config=args)

