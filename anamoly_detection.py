import numpy as np
from keras.losses import binary_crossentropy
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from datasets.stacked_mnist import StackedMNISTData, DataMode
from sae.core import AutoEncoder


def main():
    generator = StackedMNISTData(mode=DataMode.MONO_BINARY_MISSING, default_batch_size=2048)

    x_train, _ = generator.get_full_data_set(training=True)
    x_train = x_train[:, :, :, [0]]

    ae = AutoEncoder(cnn=False, file_path="models/sae_anomaly/sae_anomaly", learning_rate=0.01)
    ae.fit(
        x=x_train,
        y=x_train,
        batch_size=1024,
        epochs=20,
    )

    for datamode in [DataMode.MONO_BINARY_MISSING, DataMode.COLOR_BINARY_MISSING]:

        # generate test set
        generator = StackedMNISTData(mode=datamode, default_batch_size=2048)
        x_test, y_test = generator.get_full_data_set(training=False)

        # predict
        channels = x_test.shape[-1]
        pred = np.zeros(x_test.shape)
        for channel in range(channels):
            channel_pred = np.array(ae(x_test[:, :, :, channel]))
            pred[:, :, :, channel] = channel_pred[:, :, :, 0]

        # compute loss
        loss = []
        for y_pred, y_true in zip(pred, x_test):
            loss.append(binary_crossentropy(y_pred=y_pred.flatten(), y_true=y_true.flatten()))

        # select and display ground truth images with the greatest loss
        ind = np.argpartition(np.array(loss), -16)[-16:]
        fig = plt.figure(figsize=(4., 4.))
        grid = ImageGrid(fig, 111, nrows_ncols=(4, 4), axes_pad=0)
        for ax, img in zip(grid, x_test[ind]):
            ax.set_axis_off()
            ax.imshow(img * 255, cmap="gray")
        plt.title("Ground truth")
        plt.show()

        # display corresponding predictions
        fig = plt.figure(figsize=(4., 4.))
        grid = ImageGrid(fig, 111, nrows_ncols=(4, 4), axes_pad=0)
        for ax, img in zip(grid, pred[ind]):
            ax.set_axis_off()
            ax.imshow(img, cmap="gray")
        plt.title("Prediction")
        plt.show()


if __name__ == '__main__':
    main()
