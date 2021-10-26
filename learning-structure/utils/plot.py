import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import seaborn as sns


def plot_neurons_3d(fig, neurons: np.array, signs: np.array, show_grid=True, show_ticks=True, palette=None,
                    loc='upper right', bbox_to_anchor=(0.99, 0.9)):
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

    ax.scatter(pos_neurons[:, 0], pos_neurons[:, 1], pos_neurons[:, 2], color=pos_color,
               label='+')
    ax.scatter(neg_neurons[:, 0], neg_neurons[:, 1], neg_neurons[:, 2], color=neg_color,
               label='--')

    ax.legend(loc=loc, bbox_to_anchor=bbox_to_anchor)
    ax.grid(show_grid)
    ax._axis3don = show_ticks
    fig.add_axes(ax)
