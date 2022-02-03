import os
import numpy as np

import matplotlib as mpl
from typing import List

if os.environ.get('DISPLAY', '') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
else:
    mpl.use('TkAgg')
    mpl.rcParams['interactive'] == True

from matplotlib import pyplot as plt

from matplotlib import rc

rc('text', usetex=True)
plt.rc('font', family='serif')

plt.rcParams.update({'font.size': 30})
label_font_size = 34
title_font_size = 38
colours = mpl.cm.tab10.colors
markers = ['o', '^', 'd']
linestyles = ['-', ':', '-.', '--', '-', '--', '-.', ':']


def plot_grid_part(images: List[np.array],
                   titles: List[str] = None,
                   images_per_row: int = 3,
                   cmap: str = 'gray',
                   norm=mpl.colors.NoNorm()) -> plt.figure:
    """
    arrange images in a grid with optional titles
    """
    plt.close("all")
    num_images = len(images)
    if titles is None:
        titles = [''] * num_images
    images_per_row = min(num_images, images_per_row)

    num_rows = int(np.ceil(num_images / images_per_row))

    if len(cmap) != num_images or type(cmap) == str:
        cmap = [cmap] * num_images

    fig, axes = plt.subplots(nrows=num_rows, ncols=images_per_row)

    fig = plt.gcf()
    fig.set_size_inches(4 * images_per_row, 5 * int(np.ceil(len(images) / images_per_row)))
    for i in range(num_rows):
        for j in range(images_per_row):

            idx = images_per_row * i + j

            if num_rows == 1:
                a_ij = axes[j]
            elif images_per_row == 1:
                a_ij = axes[i]
            else:
                a_ij = axes[i, j]
            a_ij.axis('off')
            if idx >= num_images:
                break
            a_ij.imshow(images[idx], cmap=cmap[idx], norm=norm, interpolation='nearest')
            a_ij.set_title(titles[idx])

    return fig
