import click

from utils.tools import *
from utils.plot import *
from utils.nn import TARGET_FUNCS_DICT, LOSS_DICT
from networks import two_layer_net

SEED = 42
ROOT = os.path.dirname(os.path.dirname(__file__))
EXPERIMENTS_DIR = os.path.join(ROOT, 'experiments')


@click.command()
@click.option('--n_samples', '-n', required=False, type=click.INT, default=500,
              help='Input dimension')
@click.option('--input_dim', '-d', required=False, type=click.INT, default=3,
              help='Input dimension')
@click.option('--width', '-m', required=False, type=click.INT, default=1024,
              help='How many steps of SGD to take')
@click.option('--bias', '-b', required=False, type=click.BOOL, default=True,
              help='Whether to use a bias term in the network')
@click.option('--n_steps', '-N', required=False, type=click.INT, default=1000,
              help='How many steps of SGD to take')
@click.option('--base_lr', '-lr', required=False, type=click.FLOAT, default=0.01,
              help='Which learning rate to use')
@click.option('--batch_size', '-bs', required=False, type=click.INT, default=None,
              help='What batch size to use')
@click.option('--tgt_func_name', '-fn', required=False, type=click.STRING, default="target_func",
              help='Which target function to use')
@click.option('--plot_every', '-pe', required=False, type=click.INT, default=1,
              help='How often to save the neurons during SGD for later plotting')
def main(n_samples=500, input_dim=3, width=1024, bias=True, n_steps=1000, base_lr=0.01, batch_size=None,
         tgt_func_name="target_func", plot_every=1):
    if batch_size is None:
        batch_size = n_samples
    tgt_func_name = '{}_{}d'.format(tgt_func_name, input_dim)

    logger, experiment_dir = _set_up_experiment(n_samples, input_dim, width, bias, n_steps, base_lr, batch_size,
                                                tgt_func_name, plot_every)

    # data
    X = torch.randn(size=(n_samples, input_dim), requires_grad=False)
    target_func = TARGET_FUNCS_DICT[tgt_func_name]
    y = target_func(X).reshape(len(X), 1)
    logger.info("Input data shape : {}".format(X.shape))
    logger.info("Output data shape : {}".format(y.shape))

    # network
    network = two_layer_net.TwoLayerNet(input_dim=input_dim, width=width, bias=bias)
    signs = network.output_layer.weight.data.detach().numpy()[0]
    network.train()

    # loss & optimizer
    loss = LOSS_DICT['mse'](reduction='mean')
    optimizer = torch.optim.SGD(network.parameters(), lr=width * base_lr)

    try:
        neurons_trajectories = train_network(logger, X, y, network, loss, optimizer, n_steps, n_samples, batch_size,
                                             plot_every)
        try:
            plot_and_save_neuron_trajectories(logger, experiment_dir, input_dim, neurons_trajectories, signs,
                                              plot_every)
        except Exception as e:
            logger.exception("Exception while plotting the neurons' trajectories : {}".format(e))

    except Exception as e:
        logger.exception("Exception while training the network : {}".format(e))


def _set_up_experiment(input_dim, n_samples, width, bias, n_steps, base_lr, batch_size, tgt_func_name,
                       plot_every):
    set_random_seeds(SEED)
    basic_config = 'm={}_n={}_bias={}_bs={}'.format(width, n_samples, bias, batch_size)
    detailed_config = 'steps={}_lr={}_tgt-func={}_pe={}'.format(n_steps, base_lr, tgt_func_name, plot_every)
    experiment_dir = os.path.join(EXPERIMENTS_DIR, basic_config, detailed_config)
    create_dir(experiment_dir)

    logger = set_up_logger(os.path.join(experiment_dir, 'run.log'))

    _log_experiment_parameters(logger, input_dim, n_samples, width, bias, n_steps, base_lr, batch_size, tgt_func_name,
                               plot_every, experiment_dir)
    return logger, experiment_dir


def _log_experiment_parameters(logger, input_dim, n_samples, width, bias, n_steps, base_lr, batch_size, tgt_func_name,
                               plot_every, experiment_dir):
    logger.info("------ Running experiment at {} ------".format(experiment_dir))
    logger.info("input_dim : {:,}".format(input_dim))
    logger.info("n_samples : {:,}".format(n_samples))
    logger.info("width : {:,}".format(width))
    logger.info("bias : {}".format(bias))
    logger.info("n_steps : {:,}".format(n_steps))
    logger.info("base_lr : {}".format(base_lr))
    logger.info("batch_size : {:,}".format(batch_size))
    logger.info("tgt_func_name : {}".format(tgt_func_name))
    logger.info("plot_every : {:,}".format(plot_every))


def train_network(logger, X, y, network, loss, optimizer, n_steps, n_samples, batch_size, plot_every):
    neurons_trajectories = [network.first_layer.weight.data.detach().clone().numpy()]
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
        logger.info("Loss as step {:,} : {:.3f}".format(cmpt_steps, loss_.detach().item()))

        # backward and gradient step
        loss_.backward()
        optimizer.step()

        cmpt_steps += 1

        if cmpt_steps % plot_every == 0:
            neurons_trajectories.append(network.first_layer.weight.data.detach().clone().numpy())

    return neurons_trajectories


def plot_and_save_neuron_trajectories(logger, experiment_dir, input_dim, neurons, signs, plot_every):
    figures_dir = os.path.join(experiment_dir, "trajectories")
    create_dir(figures_dir)
    logger.info("Saving plots of neurons at {}".format(figures_dir))
    fig = plt.figure(figsize=(8, 8))
    if input_dim == 2:
        plot_neurons_2d(fig, neurons=neurons[0], signs=signs)
    elif input_dim == 3:
        plot_neurons_3d(fig, neurons=neurons[0], signs=signs, show_plane=True)
    fig_path = os.path.join(figures_dir, 'step_{}.png'.format(0))
    plt.savefig(fig_path, bbox_inches='tight', pad_inches=0.2)
    fig.clf()

    for i in range(1, len(neurons) + 1):
        logger.info("Plotting trajectory after {:,} steps of SGD".format(plot_every * i))
        if input_dim == 2:
            plot_neurons_trajectory_2d(fig, neurons=neurons[:i], signs=signs)
        elif input_dim == 3:
            plot_neurons_trajectory_3d(fig, neurons=neurons[:i], signs=signs, show_plane=True)
        fig_path = os.path.join(figures_dir, 'step_{}.png'.format(plot_every * i))
        plt.savefig(fig_path, bbox_inches='tight', pad_inches=0.2)
        fig.clf()


if __name__ == '__main__':
    main()
