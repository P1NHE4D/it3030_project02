from keras.models import Sequential
from keras.layers import Dense, Flatten, Reshape, Conv2D, Conv2DTranspose, MaxPooling2D
from tensorflow import keras


def init_conv_model(learning_rate=0.01):
    model = Sequential()
    model.add(Conv2D(32, input_shape=(28, 28, 1), kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu"))
    model.add(Conv2D(64, input_shape=(28, 28, 32), kernel_size=(3, 3), strides=(2, 2), padding="same", activation="relu"))
    model.add(Conv2D(64, input_shape=(14, 14, 64), kernel_size=(3, 3), strides=(2, 2), padding="same", activation="relu"))
    model.add(Conv2D(2, input_shape=(7, 7, 64), kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu"))

    model.add(Conv2DTranspose(64, input_shape=(7, 7, 2), kernel_size=(2, 2), strides=(2, 2), padding="same", activation="relu"))
    model.add(Conv2DTranspose(32, input_shape=(14, 14, 64), kernel_size=(2, 2), strides=(2, 2), padding="same", activation="relu"))
    model.add(Conv2D(1, kernel_size=(3, 3), input_shape=(28, 28, 32), strides=(1, 1), padding="same", activation="sigmoid"))
    model.compile(
        loss=keras.losses.binary_crossentropy,
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate)
    )
    return model


def init_model(learning_rate=0.01):
    model = Sequential()
    model.add(Flatten())
    model.add(Dense(256, activation="relu"))
    model.add(Dense(128, activation="relu"))
    model.add(Dense(32))
    model.add(Dense(128, activation="relu"))
    model.add(Dense(256, activation="relu"))
    model.add(Dense(784, activation="sigmoid"))
    model.add(Reshape((28, 28, 1), input_shape=(784,)))
    model.compile(
        loss=keras.losses.binary_crossentropy,
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate)
    )

    return model


class AutoEncoder:

    def __init__(self, model_type="nn", learning_rate=0.01, force_retrain=False):
        self.learning_rate = learning_rate
        self.force_retrain = force_retrain
        self.file_path = "./sae/models/sae_model"
        if model_type == "conv":
            self.model = init_conv_model(learning_rate)
        else:
            self.model = init_model(0.01)

    def load_weights(self):
        try:
            print("Weights found. Loading...")
            self.model.load_weights(filepath=self.file_path)
            return True
        except:
            print("Weights not found. Retraining...")
            return False

    def train(self, x_train, y_train, x_val, y_val, epochs=5):
        if not self.load_weights() or self.force_retrain:
            self.model.fit(x=x_train, y=y_train, batch_size=1024, epochs=epochs, validation_data=(x_val, y_val))
            self.model.save_weights(filepath=self.file_path)

    def predict(self, x):
        return self.model.predict(x)
