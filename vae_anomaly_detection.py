import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from tensorflow import keras
from tqdm import tqdm
from datasets.stacked_mnist import StackedMNISTData, DataMode
from vae.core import VariationalAutoEncoder


def main():
    # get training data
    generator = StackedMNISTData(mode=DataMode.MONO_BINARY_MISSING, default_batch_size=2048)
    x_train, _ = generator.get_full_data_set(training=True)
    x_train = x_train[:, :, :, [0]]

    # train model
    vae = VariationalAutoEncoder(
        learning_rate=0.01,
        kl_weight=0.1,
        encoded_dims=16,
        file_path="models/vae_anomaly/vae_anomaly"
    )
    vae.fit(
        x=x_train,
        y=x_train,
        batch_size=1024,
        epochs=30
    )

    for datamode in [DataMode.MONO_BINARY_MISSING, DataMode.COLOR_BINARY_MISSING]:
        # get test set
        generator = StackedMNISTData(mode=datamode, default_batch_size=2048)
        x_test, y_test = generator.get_full_data_set(training=False)

        # generate random encodings sampled from a standard normal probability distribution
        sample_size = 500
        channels = x_test.shape[-1]
        encodings = np.random.standard_normal((sample_size, 16))

        # compute decoded images using VAE
        decodings = np.array(vae.decoder(encodings).mean())

        # compute probability for image given the encodings
        probabilities = []
        for img in tqdm(x_test):
            img_prob = []
            for channel in range(channels):
                stacked_img = np.repeat([img[:, :, channel]], sample_size, axis=0)
                entropy = keras.losses.BinaryCrossentropy()(stacked_img, decodings[:, :, :, 0])
                p = np.exp(-entropy)
                img_prob.append(p)
            probabilities.append(np.mean(img_prob))
        probabilities = np.array(probabilities)

        # display the top 25 images with the lowest probability
        idx = np.argsort(probabilities)[0:25]
        fig = plt.figure(figsize=(5., 5.))
        grid = ImageGrid(fig, 111, nrows_ncols=(5, 5), axes_pad=0)
        for ax, img in zip(grid, x_test[idx]):
            ax.set_axis_off()
            ax.imshow(img * 255, cmap="gray")
        plt.show()


if __name__ == '__main__':
    main()
