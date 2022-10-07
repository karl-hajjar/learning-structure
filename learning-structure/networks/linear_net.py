import torch
from torch import nn
from pytorch_lightning import LightningModule

from utils.nn import *


class LinearNet(LightningModule):
    """
    A class defining a simple fully-connected network with two layers (one hidden layer). Input layer weights are
    initialized on the unit sphere, and output weights are sampled uniformly in {-1, 1} and the activation is :z -> z/2.
    """
    def __init__(self, input_dim: int, width: int, bias=False):
        super().__init__()
        self.input_dim = input_dim
        self.width = width
        self.bias = bias

        self._build_model()  # define input and output layer attributes
        self.initialize_parameters()  # initialize with a custom init

    def _build_model(self):
        self.input_layer = nn.Linear(in_features=self.input_dim, out_features=self.width, bias=self.bias)
        self.output_layer = nn.Linear(in_features=self.width, out_features=1, bias=self.bias)
        # self.output_layer = nn.Linear(in_features=self.width, out_features=1, bias=False)

    def initialize_parameters(self, mode='sphere_uniform'):
        if mode == 'sphere_uniform':
            input_weights = generate_uniform_sphere_weights(width=self.width, d=self.input_dim)
            output_weights = generate_bernouilli_weights(width=self.width)
            self.input_layer.weight.data.copy_(input_weights.data)
            self.output_layer.weight.data.copy_(output_weights.data)
        elif mode == 'ball_uniform':
            input_weights = generate_uniform_sphere_weights(width=self.width, d=self.input_dim)
            output_weights = generate_bernouilli_weights(width=self.width)
            rs = torch.rand(size=(self.width, self.input_dim))
            self.input_layer.weight.data.copy_(rs * input_weights.data)
            self.output_layer.weight.data.copy_(output_weights.data)
        elif mode == 'gaussian':
            input_weights = torch.normal(mean=0, std=1, size=(self.width, self.input_dim))
            output_weights = generate_bernouilli_weights(width=self.width)
            self.input_layer.weight.data.copy_(input_weights.data)
            self.output_layer.weight.data.copy_(output_weights.data)

    def forward(self, x):
        return (1 / self.width) * self.output_layer(self.input_layer.forward(x))
