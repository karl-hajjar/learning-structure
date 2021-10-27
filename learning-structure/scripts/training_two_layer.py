import torch
import click

from utils.tools import *
from utils.nn import TARGET_FUNCS_DICT, LOSS_DICT
from networks import two_layer_net

SEED = 42
PLOT_EVERY = 5


@click.command()
@click.option('--n_samples', '-n', required=False, type=click.INT, default=500,
              help='Input dimension')
@click.option('--input_dim', '-d', required=False, type=click.INT, default=3,
              help='Input dimension')
@click.option('--width', '-m', required=False, type=click.INT, default=1024,
              help='How many steps of SGD to take')
@click.option('--n_steps', '-N', required=False, type=click.INT, default=1000,
              help='How many steps of SGD to take')
@click.option('--base_lr', '-lr', required=False, type=click.FLOAT, default=0.01,
              help='Which learning rate to use')
@click.option('--batch_size', '-bs', required=False, type=click.INT, default=None,
              help='What batch size to use')
@click.option('--tgt_func_name', '-fn', required=False, type=click.STRING, default="target_func",
              help='Which target function to use')
def main(n_samples=500, input_dim=3, width=1024, n_steps=1000, base_lr=0.01, batch_size=None,
         tgt_func_name="target_func"):
    if batch_size is None:
        batch_size = n_samples

    set_random_seeds(SEED)

    # data
    X = torch.randn(size=(n_samples, input_dim), requires_grad=False)
    target_func = TARGET_FUNCS_DICT[tgt_func_name]
    y = target_func(X)

    # network
    network = two_layer_net.TwoLayerNet(input_dim=input_dim, width=width)
    network.train()

    # loss & optimizer
    loss = LOSS_DICT['mse'](reduction='mean')
    optimizer = torch.optim.SGD(network.parameters(), lr=width * base_lr)

    cmpt_steps = 0
    batch_index = 0
    while cmpt_steps < n_steps:
        if batch_index >= n_samples:
            batch_index = 0
        # batch inputs and targets
        batch_x = X[batch_index: batch_index + batch_size, :]
        batch_y = y[batch_index: batch_index + batch_size]
        batch_index += batch_index + batch_size

        # forward
        y_hat = network.forward(batch_x)
        loss_ = loss(y_hat, batch_y)

        # backward and gradient step
        loss_.backward()
        optimizer.step()

        cmpt_steps += 1


if __name__ == '__main__':
    main()
