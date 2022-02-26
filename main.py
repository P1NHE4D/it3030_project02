import numpy as np

from datasets.stacked_mnist import StackedMNISTData, DataMode
from tensorflow import keras
import matplotlib.pyplot as plt

from sae.core import AutoEncoder


def main():
    generator = StackedMNISTData(mode=DataMode.MONO_BINARY_COMPLETE, default_batch_size=2048)

    x_train, _ = generator.get_full_data_set(training=True)
    x_test, _ = generator.get_full_data_set(training=False)

    # "Translate": Only look at "red" channel; only use the last digit. Use one-hot for labels during training
    x_train = x_train[:, :, :, [0]]
    x_test = x_test[:, :, :, [0]]

    ae = AutoEncoder()
    ae.compile(
        loss=keras.losses.binary_crossentropy,
        optimizer=keras.optimizers.Adam(learning_rate=0.01)
    )
    ae.fit(
        x=x_train,
        y=x_train,
        batch_size=1024,
        epochs=5,
        validation_data=(x_test, x_test)
    )

    idx = np.random.choice(x_test.shape[0], 5)
    p = x_test[idx]
    imgs = ae.predict(p)
    for i, img in enumerate(imgs):
        plt.imshow(p[i], cmap='gray')
        plt.show()
        plt.imshow(img, cmap='gray')
        plt.show()


if __name__ == '__main__':
    main()
