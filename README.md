This repository contains an implementation of ELRG-VI, the algorithm from paper "Efficient Low Rank Gaussian Variational Inference for Neural Networks", published at Neurips conference 2020.

Link to the [paper](https://proceedings.neurips.cc/paper/2020/file/310cc7ca5a76a446f85c1a0d641ba96d-Paper.pdf).

Requires torch >= 2.0

To run vectorized MNIST experiment:

```
python vmnist.py --rank 5
```

To run small CNN experiment using LeNet:

```
python cnn.py --model LeNet --data_transform False --scale_prior 0 --dataset MNIST --prior_precision 100. --rank 5 --q_init_logvar -10 --num_updates 60000 --lr 0.001 --num_test_samples 4000
```

To run large CNN experiment using resnet18:

```
python cnn.py --dataset CIFAR10 --rank 5
```


If you find our code or paper helpful, please cite:
```
@inproceedings{NEURIPS2020_310cc7ca,
 author = {Tomczak, Marcin and Swaroop, Siddharth and Turner, Richard},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {H. Larochelle and M. Ranzato and R. Hadsell and M. F. Balcan and H. Lin},
 pages = {4610--4622},
 publisher = {Curran Associates, Inc.},
 title = {Efficient Low Rank Gaussian Variational Inference for Neural Networks},
 url = {https://proceedings.neurips.cc/paper/2020/file/310cc7ca5a76a446f85c1a0d641ba96d-Paper.pdf},
 volume = {33},
 year = {2020}}
```