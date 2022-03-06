import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from utils.verification_net import VerificationNet
from datasets.stacked_mnist import StackedMNISTData, DataMode
from vae.core import VariationalAutoEncoder


def main():
    generator = StackedMNISTData(mode=DataMode.MONO_BINARY_COMPLETE, default_batch_size=2048)
    x_train, _ = generator.get_full_data_set(training=True)
    x_train = x_train[:, :, :, [0]]

    vae = VariationalAutoEncoder(
        learning_rate=0.01,
        kl_weight=0.5,
        encoded_dims=16,
    )
    vae.fit(
        x=x_train,
        y=x_train,
        batch_size=1024,
        epochs=15
    )

    for tolerance, datamode in zip([0.8, 0.5], [DataMode.MONO_BINARY_COMPLETE, DataMode.COLOR_BINARY_COMPLETE]):

        # create test set
        generator = StackedMNISTData(mode=datamode, default_batch_size=2048)
        x_test, y_test = generator.get_full_data_set(training=False)

        # predict test set
        channels = x_test.shape[-1]
        pred = np.zeros(x_test.shape)
        for channel in range(channels):
            channel_pred = np.array(vae(x_test[:, :, :, channel]).mean())
            pred[:, :, :, channel] = channel_pred[:, :, :, 0]

        # show samples from the predicted images
        idx = np.random.choice(pred.shape[0], 8, replace=False)
        fig = plt.figure(figsize=(4., 4.))
        grid = ImageGrid(fig, 111, nrows_ncols=(4, 4), axes_pad=0)
        for ax, img in zip(grid, np.concatenate([pred[idx], x_test[idx] * 255])):
            ax.set_axis_off()
            ax.imshow(img, cmap="gray")
        plt.show()

        # run verification net on predictions to compute accuracy and predictability
        net = VerificationNet()
        cov = net.check_class_coverage(data=pred, tolerance=tolerance)
        pred, acc = net.check_predictability(data=pred, correct_labels=y_test)
        print(f"Coverage: {100 * cov:.2f}%")
        print(f"Predictability: {100 * pred:.2f}%")
        print(f"Accuracy: {100 * acc:.2f}%")

    # generative model
    rand_encoding = np.random.standard_normal((16, 16))
    decoding = np.array(vae.decoder(rand_encoding).mean())
    idx = np.random.choice(decoding.shape[0], 16, replace=False)
    fig = plt.figure(figsize=(4., 4.))
    grid = ImageGrid(fig, 111, nrows_ncols=(4, 4), axes_pad=0)
    for ax, img in zip(grid, decoding[idx]):
        ax.set_axis_off()
        ax.imshow(img, cmap="gray")
    plt.show()


if __name__ == '__main__':
    main()
