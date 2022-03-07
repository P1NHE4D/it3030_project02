from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Reshape, Conv2D, Conv2DTranspose, MaxPooling2D, Dropout
from tensorflow import keras


def construct_encoder(cnn, encoded_dims):
    if cnn:
        encoder = Sequential([
            Conv2D(16, input_shape=(28, 28, 1), kernel_size=(3, 3), strides=(2, 2), padding="same", activation="leaky_relu"),
            Conv2D(32, kernel_size=(3, 3), strides=(2, 2), padding="same", activation="leaky_relu"),
            Conv2D(1, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="leaky_relu"),
            Flatten(),
            Dense(encoded_dims, activation="leaky_relu")
        ])
    else:
        encoder = Sequential([
            Flatten(),
            Dense(256, activation="leaky_relu"),
            Dense(128, activation="leaky_relu"),
            Dense(encoded_dims, activation="leaky_relu"),
        ])
    return encoder


def construct_decoder(cnn):
    if cnn:
        decoder = Sequential([
            Dense(49, activation="leaky_relu"),
            Reshape((7, 7, 1)),
            Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="leaky_relu"),
            Conv2DTranspose(32, kernel_size=(3, 3), strides=(2, 2), padding="same", activation="leaky_relu"),
            Conv2D(16, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="leaky_relu"),
            Conv2DTranspose(16, kernel_size=(3, 3), strides=(2, 2), padding="same", activation="leaky_relu"),
            Conv2D(1, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="sigmoid"),
        ])
    else:
        decoder = Sequential([
            Dense(128, activation="leaky_relu"),
            Dense(256, activation="leaky_relu"),
            Dense(784, activation="sigmoid"),
            Reshape((28, 28, 1), input_shape=(784,)),
        ])
    return decoder


class AutoEncoder(Model):

    def __init__(self, learning_rate=0.01, encoded_dims=16, file_path="models/sae/sae", retrain=False, cnn=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.retrain = retrain
        self.file_path = file_path
        self.cnn = cnn
        self.encoded_dims = encoded_dims
        self.encoder = construct_encoder(self.cnn, self.encoded_dims)
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

    def get_config(self):
        super().get_config()
