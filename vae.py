import numpy as np
from matplotlib import pyplot as plt
from utils.verification_net import VerificationNet

from datasets.stacked_mnist import StackedMNISTData, DataMode
from vae.core import VariationalAutoEncoder


def main():
    generator = StackedMNISTData(mode=DataMode.MONO_BINARY_COMPLETE, default_batch_size=2048)

    x_train, _ = generator.get_full_data_set(training=True)
    x_test, y_test = generator.get_full_data_set(training=False)

    vae = VariationalAutoEncoder(
        learning_rate=0.01,
        kl_weight=1,
        encoded_dims=16,
        retrain=False
    )
    vae.fit(
        x=x_train[:, :, :, [0]],
        y=x_train[:, :, :, [0]],
        batch_size=1024,
        epochs=15
    )

    pred = vae(x_test).mean()
    pred = np.array(pred)

    net = VerificationNet()
    # img, labels = generator.get_random_batch(training=False, batch_size=25000)
    cov = net.check_class_coverage(data=pred, tolerance=.8)
    pred, acc = net.check_predictability(data=pred, correct_labels=y_test)
    print(f"Coverage: {100 * cov:.2f}%")
    print(f"Predictability: {100 * pred:.2f}%")
    print(f"Accuracy: {100 * acc:.2f}%")


    # idx = np.random.choice(x_test.shape[0], 10)
    # p = x_test[idx]
    # imgs = vae(p).mean()
    # for i, img in enumerate(imgs):
    #     plt.imshow(p[i], cmap="gray")
    #     plt.show()
    #     plt.imshow(img, cmap="gray")
    #     plt.show()

    # generative model
    # rand_encoding = np.random.standard_normal((16, 1, 1, 16))
    # decoding = vae.decoder(rand_encoding).mean()
    # f, axs = plt.subplots(4, 4)
    # row = 0
    # col = 0
    # for img in decoding:
    #     axs[row, col].set_axis_off()
    #     axs[row, col].imshow(img, cmap='gray')
    #     col = (col + 1) % 4
    #     if col == 0:
    #         row += 1
    # plt.subplots_adjust(wspace=0, hspace=0)
    # plt.show()


if __name__ == '__main__':
    main()
