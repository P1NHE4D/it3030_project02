from keras.models import Sequential
from keras.layers import Dense, Flatten, Reshape
from tensorflow import keras


def init_model(learning_rate=0.01):
    model = Sequential()
    model.add(Flatten())
    model.add(Dense(256, activation="relu"))
    model.add(Dense(128, activation="relu"))
    model.add(Dense(32))
    model.add(Dense(128, activation="relu"))
    model.add(Dense(256, activation="relu"))
    model.add(Dense(784, activation="sigmoid"))
    model.add(Reshape((28, 28, 1), input_shape=(784, )))
    model.compile(
        loss=keras.losses.binary_crossentropy,
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate)
    )

    return model


class AutoEncoder:

    def __init__(self):
        self.model = init_model(0.01)

    def train(self, x_train, y_train, x_val, y_val, epochs=5):
        self.model.fit(x=x_train, y=y_train, batch_size=1024, epochs=epochs, validation_data=(x_val, y_val))

    def predict(self, x):
        return self.model.predict(x)
