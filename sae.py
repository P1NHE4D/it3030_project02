import numpy as np
from datasets.stacked_mnist import StackedMNISTData, DataMode
import matplotlib.pyplot as plt
from sae.core import AutoEncoder
from utils.plot import show_image_grid
from utils.verification_net import VerificationNet


def main():
    generator = StackedMNISTData(mode=DataMode.COLOR_BINARY_COMPLETE, default_batch_size=2048)

    x_train, _ = generator.get_full_data_set(training=True)
    x_test, y_test = generator.get_full_data_set(training=False)

    ae = AutoEncoder(retrain=False, cnn=True, file_path="models/sae_cnn/sae_cnn")
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

    show_image_grid(pred)

    # verification
    net = VerificationNet()
    cov = net.check_class_coverage(data=x_test, tolerance=.5)
    pred, acc = net.check_predictability(data=x_test, correct_labels=y_test)
    print(f"Coverage: {100 * cov:.2f}%")
    print(f"Predictability: {100 * pred:.2f}%")
    print(f"Accuracy: {100 * acc:.2f}%")

    # generative model
    rand_encoding = np.random.randn(16, 7, 7)
    decoding = np.array(ae.decoder(rand_encoding))
    show_image_grid(decoding)


if __name__ == '__main__':
    main()
