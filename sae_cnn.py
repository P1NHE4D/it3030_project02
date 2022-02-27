import numpy as np
from datasets.stacked_mnist import StackedMNISTData, DataMode
import matplotlib.pyplot as plt
from sae.core import AutoEncoder


def main():
    generator = StackedMNISTData(mode=DataMode.MONO_BINARY_COMPLETE, default_batch_size=2048)

    x_train, _ = generator.get_full_data_set(training=True)
    x_test, _ = generator.get_full_data_set(training=False)

    ae = AutoEncoder(retrain=False, cnn=True, file_path="models/sae_cnn/sae_cnn")
    ae.fit(
        x=x_train[:, :, :, [0]],
        y=x_train[:, :, :, [0]],
        batch_size=1024,
        epochs=5
    )

    idx = np.random.choice(x_test.shape[0], 1)
    p = x_test[idx]
    imgs = ae.predict(p)
    for i, img in enumerate(imgs):
        plt.imshow(p[i])
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
