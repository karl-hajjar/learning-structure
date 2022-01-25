import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import seaborn as sns


def plot_neurons_3d(fig, neurons: np.array, signs: np.array, show_grid=True, show_ticks=True, palette=None,
                    loc='upper right', bbox_to_anchor=(0.99, 0.9), show_plane=True, num=10):
    if palette is None:
        palette = sns.color_palette()
    else:
        if len(palette) < 2:
            raise ValueError("Color palette has less than 2 colours.")

    pos_neurons = neurons[signs > 0, :]
    neg_neurons = neurons[signs < 0, :]

    pos_color = palette[0]
    neg_color = palette[1]

    ax = Axes3D(fig, auto_add_to_figure=False)

    ax.scatter(pos_neurons[:, 0], pos_neurons[:, 1], pos_neurons[:, 2], color=pos_color, label='+')
    ax.scatter(neg_neurons[:, 0], neg_neurons[:, 1], neg_neurons[:, 2], color=neg_color, label='--')

    x_min, x_max = ax.get_xlim3d()
    y_min, y_max = ax.get_ylim3d()

    if show_plane:
        xs = np.linspace(x_min, x_max, num=num, endpoint=True)
        ys = np.linspace(y_min, y_max, num=num, endpoint=True)
        X, Y = np.meshgrid(xs, ys)
        Z = np.zeros_like(X)

        ax.plot_wireframe(X, Y, Z, color='r', linewidth=0.5)

    ax.legend(loc=loc, bbox_to_anchor=bbox_to_anchor)
    ax.grid(show_grid)
    ax._axis3don = show_ticks
    fig.add_axes(ax)


def plot_neurons_trajectory_3d(fig, neurons: [list, np.array], signs: np.array, show_grid=True, show_ticks=True,
                               palette=None, loc='upper right', bbox_to_anchor=(0.99, 0.9), linewidth=0.04,
                               show_plane=True, num=10):
    if palette is None:
        palette = sns.color_palette()
    else:
        if len(palette) < 2:
            raise ValueError("Color palette has less than 2 colours.")

    neurons_ = neurons[:len(neurons)-1]
    neurons = neurons[len(neurons)-1]
    trajectories = [[all_neurons[j, :] for all_neurons in neurons_] for j in range(len(neurons))]

    pos_color = palette[0]
    neg_color = palette[1]
    ax = Axes3D(fig, auto_add_to_figure=False)

    # trajectories
    if len(neurons_) > 0:
        for trajectory in trajectories:
            ax.plot([neuron[0] for neuron in trajectory], [neuron[1] for neuron in trajectory],
                    [neuron[2] for neuron in trajectory], color='k', linewidth=linewidth)

    pos_neurons = neurons[signs > 0, :]
    neg_neurons = neurons[signs < 0, :]

    # final positions at time t = len(neurons)
    ax.scatter(pos_neurons[:, 0], pos_neurons[:, 1], pos_neurons[:, 2], color=pos_color, label='+')
    ax.scatter(neg_neurons[:, 0], neg_neurons[:, 1], neg_neurons[:, 2], color=neg_color, label='--')
    x_min, x_max = ax.get_xlim3d()
    y_min, y_max = ax.get_ylim3d()

    if show_plane:
        xs = np.linspace(x_min, x_max, num=num, endpoint=True)
        ys = np.linspace(y_min, y_max, num=num, endpoint=True)
        X, Y = np.meshgrid(xs, ys)
        Z = np.zeros_like(X)

        ax.plot_wireframe(X, Y, Z, color='r', linewidth=0.5)

    ax.legend(loc=loc, bbox_to_anchor=bbox_to_anchor)
    ax.grid(show_grid)
    ax._axis3don = show_ticks
    fig.add_axes(ax)


def plot_neurons_2d(fig, neurons: np.array, signs: np.array, show_grid=True, show_ticks=True, palette=None,
                    loc='upper left', bbox_to_anchor=(1.01, 0.95), s=10):
    if palette is None:
        palette = sns.color_palette()
    else:
        if len(palette) < 2:
            raise ValueError("Color palette has less than 2 colours.")

    pos_neurons = neurons[signs > 0, :]
    neg_neurons = neurons[signs < 0, :]

    pos_color = palette[0]
    neg_color = palette[1]

    ax = fig.gca()

    ax.scatter(pos_neurons[:, 0], pos_neurons[:, 1], color=pos_color, label='+', s=s)
    ax.scatter(neg_neurons[:, 0], neg_neurons[:, 1], color=neg_color, label='--', s=s)

    ax.legend(loc=loc, bbox_to_anchor=bbox_to_anchor)
    ax.grid(show_grid)
    if not show_ticks:
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])


def plot_neurons_trajectory_2d(fig, neurons: np.array, signs: np.array, show_grid=True, show_ticks=True,
                               palette=None, linewidth=0.04, loc='upper left', bbox_to_anchor=(1.01, 0.95), s=10,
                               skip=0):
    if palette is None:
        palette = sns.color_palette()
    else:
        if len(palette) < 2:
            raise ValueError("Color palette has less than 2 colours.")

    x_max = 1.1 * np.max(np.abs(neurons[:, :, 0]))
    y_max = 1.1 * np.max(np.abs(neurons[:, :, 1]))

    pos_color = palette[0]
    neg_color = palette[1]
    ax = fig.gca()
    ax.set_xlim(xmin=-x_max, xmax=x_max)
    ax.set_ylim(ymin=-y_max, ymax=y_max)

    last_neurons = neurons[-1, :, :]  # neurons' shape is (T, m, d)
    neurons = neurons[:neurons.shape[0]-1, :, :]

    # sub-sample neurons according to the value of skip
    neurons = np.array([neurons[t, :, :] for t in range(neurons.shape[0]) if (t % (skip + 1) == 0)])

    # trajectories
    if neurons.shape[0] > 0:
        for j in range(neurons.shape[1]):  # for neuron at index j, plot its trajectory over time
            ax.plot(neurons[:, j, 0], neurons[:, j, 1], color='k', linewidth=linewidth)

    pos_neurons = last_neurons[signs > 0, :]
    neg_neurons = last_neurons[signs < 0, :]

    # final positions at time T = len(neurons)
    plt.xlim(-x_max, x_max)
    plt.ylim(-y_max, y_max)
    plt.scatter(pos_neurons[:, 0], pos_neurons[:, 1], color=pos_color, label='+', s=s)
    plt.scatter(neg_neurons[:, 0], neg_neurons[:, 1], color=neg_color, label='--', s=s)

    plt.legend(loc=loc, bbox_to_anchor=bbox_to_anchor)
    plt.grid(show_grid)
    if not show_ticks:
        plt.xticks([])
        plt.yticks([])
