import numpy as np
from matplotlib import pyplot as plt

from datasets.stacked_mnist import StackedMNISTData, DataMode
from vae.core import VariationalAutoEncoder


def main():
    generator = StackedMNISTData(mode=DataMode.MONO_BINARY_COMPLETE, default_batch_size=2048)

    x_train, _ = generator.get_full_data_set(training=True)
    x_test, _ = generator.get_full_data_set(training=False)

    vae = VariationalAutoEncoder(kl_weight=1, retrain=True, learning_rate=0.01, encoded_dims=50)
    vae.fit(
        x=x_train[:, :, :, [0]],
        y=x_train[:, :, :, [0]],
        batch_size=1024,
        epochs=5
    )

    idx = np.random.choice(x_test.shape[0], 1)
    p = x_test[idx]
    imgs = vae.predict(p)
    for i, img in enumerate(imgs):
        plt.imshow(p[i], cmap="gray")
        plt.show()
        plt.imshow(img, cmap="gray")
        plt.show()


if __name__ == '__main__':
    main()
