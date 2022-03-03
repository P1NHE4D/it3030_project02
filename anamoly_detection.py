import numpy as np
from keras.losses import binary_crossentropy
from matplotlib import pyplot as plt
from datasets.stacked_mnist import StackedMNISTData, DataMode
from sae.core import AutoEncoder


def main():
    generator = StackedMNISTData(mode=DataMode.COLOR_BINARY_MISSING, default_batch_size=2048)

    x_train, _ = generator.get_full_data_set(training=True)
    x_test, _ = generator.get_full_data_set(training=False)

    ae = AutoEncoder(cnn=True, file_path="models/sae_anomaly/sae_anomaly", learning_rate=0.01)
    ae.fit(
        x=x_train[:, :, :, [0]],
        y=x_train[:, :, :, [0]],
        batch_size=1024,
        epochs=5
    )

    channels = x_test.shape[-1]
    pred = np.zeros(x_test.shape)
    for channel in range(channels):
        channel_pred = np.array(ae(x_test[:, :, :, channel]))
        pred[:, :, :, channel] = channel_pred[:, :, :, 0]

    loss = []
    for y_pred, y_true in zip(pred, x_test):
        loss.append(binary_crossentropy(y_pred=y_pred.flatten(), y_true=y_true.flatten()))
    ind = np.argpartition(np.array(loss), -10)[-10:]
    for img in pred[ind]:
        plt.imshow(img, cmap="gray")
        plt.show()


if __name__ == '__main__':
    main()
