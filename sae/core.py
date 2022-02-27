from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Reshape, Conv2D, UpSampling2D, MaxPooling2D, Conv2DTranspose
from matplotlib import pyplot as plt
from tensorflow import keras


def construct_encoder(cnn):
    if cnn:
        encoder = Sequential([
            Conv2D(16, input_shape=(28, 28, 1), kernel_size=(3, 3), strides=(2, 2), padding="same", activation="relu"),
            Conv2D(32, kernel_size=(3, 3), strides=(2, 2), padding="same", activation="relu"),
            Conv2D(1, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu"),
        ])
    else:
        encoder = Sequential([
            Flatten(),
            Dense(256, activation="relu"),
            Dense(128, activation="relu"),
            Dense(32, activation="relu"),
        ])
    return encoder


def construct_decoder(cnn):
    if cnn:
        decoder = Sequential([
            Conv2DTranspose(32, kernel_size=(3, 3), strides=(2, 2), padding="same", activation="relu"),
            Conv2DTranspose(16, kernel_size=(3, 3), strides=(2, 2), padding="same", activation="relu"),
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

    def __init__(self, learning_rate=0.01, file_path="/Users/agerlach/uni_dev/it3030_project02/sae/models/sae_model", retrain=False, cnn=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.retrain = retrain
        self.file_path = file_path
        self.cnn = cnn
        self.encoder = construct_encoder(self.cnn)
        self.decoder = construct_decoder(self.cnn)
        self.compile(
            loss=keras.losses.binary_crossentropy,
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate)
        )
        try:
            self.load_weights(filepath=self.file_path)
            self.model_trained = True
        except:
            print("No predefined weights found")
            self.model_trained = False

    def call(self, inputs, **kwargs):
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        return decoded

    # noinspection PyBroadException
    def fit(self, **kwargs):
        if not self.model_trained or self.retrain:
            super().fit(**kwargs)
            print("Storing learned weights...")
            self.save_weights(self.file_path)
            self.model_trained = True

    def predict(self, x, **kwargs):
        if not self.model_trained:
            raise Exception("No weights found. Model needs to be trained first.")
        x_count = x.shape[0]
        channels = x.shape[3]
        reshaped = False
        if channels > 1:
            reshaped = True
            x = x.transpose(0, 3, 1, 2).reshape((x_count * channels, x.shape[1], x.shape[2], 1))
        pred = super().predict(x, **kwargs)
        if reshaped:
            # TODO: reshape not correct
            pred = pred.T.reshape((x_count, pred.shape[1], pred.shape[2], channels))
        return pred

    def get_config(self):
        super().get_config()
