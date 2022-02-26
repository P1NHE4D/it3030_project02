from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Reshape, Conv2D, UpSampling2D


def construct_encoder(cnn):
    if cnn:
        encoder = Sequential([
            Conv2D(16, input_shape=(28, 28, 1), kernel_size=(3, 3), strides=(2, 2), padding="same", activation="relu"),
            Conv2D(32, kernel_size=(3, 3), strides=(2, 2), padding="same", activation="relu"),
            Conv2D(1, kernel_size=(3, 3), strides=(1, 1), padding="same"),
        ])
    else:
        encoder = Sequential([
            Flatten(),
            Dense(256, activation="relu"),
            Dense(128, activation="relu"),
            Dense(32),
        ])
    return encoder


def construct_decoder(cnn):
    if cnn:
        decoder = Sequential([
            Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu"),
            UpSampling2D(),
            Conv2D(16, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu"),
            UpSampling2D(),
            Conv2D(1, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="sigmoid"),
        ])
    else:
        decoder = Sequential([
            Dense(128, activation="relu"),
            Dense(256, activation="relu"),
            Dense(784, activation="sigmoid"),
            Reshape((28, 28, 1), input_shape=(784,)),
        ])
    return decoder


class AutoEncoder(Model):

    def __init__(self, file_path="sae/models/sae_model", retrain=False, cnn=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.retrain = retrain
        self.file_path = file_path
        self.cnn = cnn
        self.encoder = construct_encoder(self.cnn)
        self.decoder = construct_decoder(self.cnn)

    def call(self, inputs, **kwargs):
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        return decoded

    # noinspection PyBroadException
    def fit(self, x, y, validation_data, **kwargs):
        try:
            self.load_weights(filepath=self.file_path)
            weights_loaded = True
        except:
            print("Weights not found. Retraining...")
            weights_loaded = False
        if not weights_loaded or self.retrain:
            channels = x.shape[3]
            if channels > 1:
                x = x[:, :, :, [0]]
                y = y[:, :, :, [0]]
                validation_data = (validation_data[0][:, :, :, [0]], validation_data[1][:, :, :, [0]])
            super().fit(x=x, y=y, validation_data=validation_data, **kwargs)
            print("Storing learned weights...")
            self.save_weights(self.file_path)

    def predict(self, x, **kwargs):
        x_count = x.shape[0]
        channels = x.shape[3]
        reshaped = False
        if channels > 1:
            reshaped = True
            x = x.T.reshape((x_count * channels, x.shape[1], x.shape[2], 1))
        pred = super().predict(x, **kwargs)
        if reshaped:
            pred = pred.T.reshape((x_count, pred.shape[1], pred.shape[2], channels))
        return pred

    def get_config(self):
        super().get_config()
