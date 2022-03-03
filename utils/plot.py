import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import ImageGrid


def show_image_grid(pred):
    idx = np.random.choice(pred.shape[0], 16, replace=False)
    fig = plt.figure(figsize=(4., 4.))
    grid = ImageGrid(fig, 111, nrows_ncols=(4, 4), axes_pad=0)
    for ax, img in zip(grid, pred[idx]):
        ax.set_axis_off()
        ax.imshow(img, cmap="gray")
    plt.show()


def show_images(pred, gt=None, sample_size=5):
    idx = np.random.choice(pred.shape[0], sample_size, replace=False)
    for i in idx:
        plt.imshow(pred[i], cmap="gray")
        plt.show()
        if gt is not None:
            plt.imshow(gt[i], cmap="gray")
            plt.show()

