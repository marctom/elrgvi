import argparse
import logging
from pathlib import Path
import time
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms as T
import torchvision

from alg import elrg
from fit import fit, DL

logging.basicConfig(
    format="[%(asctime)s, %(levelname)s] %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
    level=logging.INFO,
)

torch.backends.cudnn.benchmark = True


def fully_connected(in_dim, out_dim, num_layers, size):
    layers = [nn.Linear(in_dim, size), nn.ReLU(inplace=True)]
    for _ in range(num_layers - 1):
        layers.extend([nn.Linear(size, size), nn.ReLU(inplace=True)])
    layers.append(nn.Linear(size, out_dim))
    return nn.Sequential(*layers)


def train_vectorized_mnist(config):
    model = fully_connected(in_dim=784,
                            out_dim=10,
                            num_layers=config.num_hidden_layers,
                            size=config.num_hidden_units)
    model = nn.Sequential(model,
                          nn.LogSoftmax(dim=-1))

    def get_data(download):
        try:
            tfms = T.Compose([T.ToTensor(), T.Lambda(lambda x: torch.flatten(x))])
            train_data = torchvision.datasets.MNIST(
                '/tmp',
                download=download,
                train=True,
                transform=tfms,
            )

            test_data = torchvision.datasets.MNIST(
                '/tmp',
                download=download,
                train=False,
                transform=tfms,
            )
        except (RuntimeError, ValueError, EOFError):
            time.sleep(int(10*random.random()))
            return get_data(download)

        return train_data, test_data
    
    train_data, test_data = get_data(download=True)

    num_data = len(train_data.data)
    num_test_data = len(test_data.data)

    x = (train_data.data / 255. - 0.5).mul(2).view(num_data, -1)
    x_test = (test_data.data / 255. - 0.5).mul(2).view(num_test_data, -1)


    data = SimpleNamespace(train_dl=DL(x,
                                       train_data.targets,
                                       config.batch_size,
                                       device = config.device),
                           valid_dl=DL(x_test,
                                       test_data.targets,
                                       config.test_batch_size,
                                       device = config.device)
                           )
    
    model = elrg(model, config, num_data)
    opt = torch.optim.Adam(model.parameters(),
                           lr=config.lr)

    model.to(config.device)

    results = fit(
        model=model,
        data=data,
        loss_func=F.nll_loss,
        num_updates=config.num_updates,
        keep_curve=True,
        device=config.device,
        log_steps=500,
        eval_log_steps=20000,
    )


    logging.info(
        f"Final test nll {results.loss:.3f} "
        f"error rate {results.er:.2f}"
    )

if __name__ == "__main__":
    
    def str2bool(v):
        return v.lower() in ("yes", "true", "t", "1")
        
    parser = argparse.ArgumentParser()

    parser.add_argument("--num_hidden_layers",
                        type=int,
                        default=2)
    parser.add_argument("--num_hidden_units",
                        type=int,
                        default=400)
    parser.add_argument("--rank",
                        type=int,
                        default=1)
    parser.add_argument("--lr",
                    type=float,
                    default=0.0003)
    parser.add_argument("--learn_diag",
                        type=str2bool,
                        default=False)
    parser.add_argument("--num_updates",
                        type=int,
                        default=100000)
    parser.add_argument("--batch_size",
                        type=int,
                        default=2048)
    parser.add_argument("--device",
                        type=str,
                       default="cuda")
    parser.add_argument("--num_test_samples",
                        type=int,
                        default=2000, )
    parser.add_argument("--test_batch_size",
                        type=int,
                        default=256, )
    parser.add_argument("--scale_prior",
                        help="Scale prior variances by the size of the input to the layer (Radford Neal).",
                        type=str2bool,
                        default=True)
    parser.add_argument("--q_init_logvar",
                        help="Initial log variance of mean-field "
                             "Gaussian variational posterior.",
                        type=float,
                        default=-10, )
    parser.add_argument("--prior_precision",
                        help="Precision of prior over weights.",
                        type=float,
                        default=0.01, )

    args = parser.parse_args()
    logging.info(f"Config: {args}")
    train_vectorized_mnist(config=args)
