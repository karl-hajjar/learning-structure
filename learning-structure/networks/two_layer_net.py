from torch import nn
from pytorch_lightning import LightningModule

from utils.nn import *


class TwoLayerNet(LightningModule):
    """
    A class defining a simple fully-connected network with two layers (one hidden layer). Input layer weights are
    initialized on the unit sphere, and output weights are sampled uniformly in {-1, 1}.
    """
    def __init__(self, input_dim: int, width: int, activation: [str, None] = None, bias=False, **kwargs):
        super().__init__()
        self.input_dim = input_dim
        self.width = width
        if activation is None:
            self.activation_name = DEFAULT_ACTIVATION
        else:
            self.activation_name = activation
        self.activation = ACTIVATION_DICT[self.activation_name](**kwargs)
        self.bias = bias

        self._build_model()  # define input and output layer attributes
        self.initialize_parameters()  # initialize with a custom init

    def _build_model(self):
        self.input_layer = nn.Linear(in_features=self.input_dim, out_features=self.width, bias=self.bias)
        self.output_layer = nn.Linear(in_features=self.width, out_features=1, bias=self.bias)
        # self.output_layer = nn.Linear(in_features=self.width, out_features=1, bias=False)

    def initialize_parameters(self):
        input_weights = generate_uniform_sphere_weights(width=self.width, d=self.input_dim)
        output_weights = generate_bernouilli_weights(width=self.width)
        self.input_layer.weight.data.copy_(input_weights.data)
        self.output_layer.weight.data.copy_(output_weights.data)

    def forward(self, x):
        x = self.activation(self.input_layer.forward(x))
        return (1 / self.width) * self.output_layer(x)
