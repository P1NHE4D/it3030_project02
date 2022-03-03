import numpy as np
from datasets.stacked_mnist import StackedMNISTData, DataMode
from sae.core import AutoEncoder
from utils.plot import show_image_grid
from utils.verification_net import VerificationNet


def main():
    generator = StackedMNISTData(mode=DataMode.COLOR_BINARY_COMPLETE, default_batch_size=2048)

    x, _ = generator.get_full_data_set(training=True)
    val_idx = np.random.choice(x.shape[0], round(x.shape[0] * 0.2), replace=False)
    train_idx = np.setdiff1d(np.arange(start=0, stop=x.shape[0]), val_idx)
    x_train = x[train_idx]
    x_val = x[val_idx]

    ae = AutoEncoder(retrain=True, cnn=True, file_path="models/sae_cnn/sae_cnn")
    ae.fit(
        x=x_train[:, :, :, [0]],
        y=x_train[:, :, :, [0]],
        batch_size=1024,
        epochs=5,
        validation_data=(x_val[:, :, :, [0]], x_val[:, :, :, [0]])
    )

    for tolerance, datamode in zip([0.8, 0.5], [DataMode.MONO_BINARY_COMPLETE, DataMode.COLOR_BINARY_COMPLETE]):

        # create test set
        generator = StackedMNISTData(mode=datamode, default_batch_size=2048)
        x_test, y_test = generator.get_full_data_set(training=False)

        # predict test set
        channels = x_test.shape[-1]
        pred = np.zeros(x_test.shape)
        for channel in range(channels):
            channel_pred = np.array(ae(x_test[:, :, :, channel]))
            pred[:, :, :, channel] = channel_pred[:, :, :, 0]

        # show samples from the predicted images
        show_image_grid(pred)

        # run verification net on predictions to compute accuracy and predictability
        net = VerificationNet()
        cov = net.check_class_coverage(data=pred, tolerance=tolerance)
        pred, acc = net.check_predictability(data=pred, correct_labels=y_test)
        print(f"Coverage: {100 * cov:.2f}%")
        print(f"Predictability: {100 * pred:.2f}%")
        print(f"Accuracy: {100 * acc:.2f}%")

    # generative model
    rand_encoding = np.random.randn(16, 7, 7)
    decoding = np.array(ae.decoder(rand_encoding))
    show_image_grid(decoding)


if __name__ == '__main__':
    main()
