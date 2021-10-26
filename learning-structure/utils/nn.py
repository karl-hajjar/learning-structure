import torch


ACTIVATION_DICT = {'relu': torch.nn.ReLU,
                   'elu': torch.nn.ELU,
                   'gelu': torch.nn.GELU,
                   'sigmoid': torch.nn.modules.activation.Sigmoid,
                   'tanh': torch.nn.modules.activation.Tanh,
                   'leaky_relu': torch.nn.modules.activation.LeakyReLU,
                   'identity': torch.nn.Identity}
DEFAULT_ACTIVATION = "relu"

LOSS_DICT = {'cross_entropy': torch.nn.CrossEntropyLoss,
             'kl': torch.nn.KLDivLoss,
             'mse': torch.nn.MSELoss,
             'logistic': torch.nn.BCEWithLogitsLoss}

DEFAULT_LOSS = "cross_entropy"


def generate_uniform_sphere_weights(width: int, d: int) -> torch.Tensor:
    """
    Generates `width` iid (input) neurons in R^d uniformly sampled on the unit sphere
    :param width: number of neurons in the layer
    :param d: dimension of the data of the weights
    :return: a tensor with width rows and d columns containing the neurons
    """
    x = torch.randn(size=(width, d), requires_grad=False)
    row_norms = torch.linalg.vector_norm(x, dim=1)
    return x / row_norms.reshape(x.shape[0], 1)


def generate_bernouilli_weights(width: int) -> torch.Tensor:
    """
    Generates one (output) neuron with `width` coordinates independently sampled from {-1,1}
    :param width:
    :return: a tensor with one row and `width` columns containing the neuron
    """
    return 2 * torch.randint(low=0, high=2, size=(width,), requires_grad=False) - 1


def target_func(x: torch.Tensor) -> torch.Tensor:
    return x[:, 0].abs() + x[:, 1]


def even_target_func(x: torch.Tensor) -> torch.Tensor:
    return x[:, 0].abs() + x[:, 1].abs()


def odd_target_func(x: torch.Tensor) -> torch.Tensor:
    return x[:, 0] + x[:, 1]


TARGET_FUNCS_DICT = {'target_func': target_func,
                     'even_target_func': even_target_func,
                     'odd_target_func': odd_target_func}
