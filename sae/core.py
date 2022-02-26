from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Reshape


class AutoEncoder(Model):

    def __init__(self, file_path="sae/models/sae_model", retrain=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.retrain = retrain
        self.file_path = file_path
        self.encoder = Sequential([
            Flatten(),
            Dense(256, activation="relu"),
            Dense(128, activation="relu"),
            Dense(32),
        ])
        self.decoder = Sequential([
            Dense(128, activation="relu"),
            Dense(256, activation="relu"),
            Dense(784, activation="sigmoid"),
            Reshape((28, 28, 1), input_shape=(784,)),
        ])

    def call(self, inputs, **kwargs):
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        return decoded

    # noinspection PyBroadException
    def fit(self, **kwargs):
        try:
            self.load_weights(filepath=self.file_path)
            weights_loaded = True
        except:
            print("Weights not found. Retraining...")
            weights_loaded = False
        if not weights_loaded or self.retrain:
            super().fit(**kwargs)
            print("Storing learned weights...")
            self.save_weights(self.file_path)

    def get_config(self):
        super().get_config()
