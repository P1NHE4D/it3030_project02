import numpy as np

from datasets.stacked_mnist import StackedMNISTData, DataMode
from tensorflow import keras
import matplotlib.pyplot as plt

from sae.core import AutoEncoder


def main():
    generator = StackedMNISTData(mode=DataMode.COLOR_BINARY_COMPLETE, default_batch_size=2048)

    x_train, _ = generator.get_full_data_set(training=True)
    x_test, _ = generator.get_full_data_set(training=False)

    ae = AutoEncoder(retrain=False, cnn=True)
    ae.fit(
        x=x_train,
        y=x_train,
        batch_size=1024,
        epochs=5,
        validation_data=(x_test, x_test)
    )

    idx = np.random.choice(x_test.shape[0], 1)
    p = x_test[idx]
    imgs = ae.predict(p)
    for i, img in enumerate(imgs):
        img *= 255
        plt.imshow(p[i] * 255)
        plt.show()
        plt.imshow(img)
        plt.show()

    # generative model
    rand_encoding = np.random.randn(16, 7, 7)
    decoding = ae.decoder.predict(rand_encoding)

    f, axs = plt.subplots(4, 4)
    row = 0
    col = 0
    for img in decoding:
        axs[row, col].set_axis_off()
        axs[row, col].imshow(img, cmap='gray')
        col = (col + 1) % 4
        if col == 0:
            row += 1
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()


if __name__ == '__main__':
    main()
