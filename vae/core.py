from tensorflow_probability import layers as tfpl
from tensorflow_probability import distributions as tfpd
from keras.models import Model, Sequential
from keras.layers import Dense, Flatten, Conv2D, Conv2DTranspose, Reshape
import tensorflow as tf
from tensorflow import keras


class VariationalAutoEncoder(Model):

    def __init__(
            self,
            learning_rate=0.01,
            kl_weight=1,
            encoded_dims=4,
            retrain=False,
            file_path="models/vae/vae",
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.file_path = file_path
        self.learning_rate = learning_rate
        self.kl_weight = kl_weight
        self.encoded_dims = encoded_dims
        self.encoder = None
        self.decoder = None
        self.model_trained = False
        self.retrain = retrain
        self.setup_model()

    def setup_model(self):
        prior = tfpd.Independent(
            tfpd.Normal(
                loc=tf.zeros(self.encoded_dims),
                scale=1
            ),
            reinterpreted_batch_ndims=1
        )
        self.encoder = Sequential([
            Conv2D(16, input_shape=(28, 28, 1), kernel_size=(3, 3), strides=(2, 2), padding="same", activation="relu"),
            # Conv2D(16, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu"),
            Conv2D(32, kernel_size=(3, 3), strides=(2, 2), padding="same", activation="relu"),
            # Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu"),
            Flatten(),
            Dense(tfpl.IndependentNormal.params_size(self.encoded_dims)),
            tfpl.IndependentNormal(
                self.encoded_dims,
                convert_to_tensor_fn=tfpd.Distribution.sample,
                activity_regularizer=tfpl.KLDivergenceRegularizer(prior, weight=self.kl_weight))
        ])

        self.decoder = Sequential([
            Dense(7 * 7 * self.encoded_dims),
            Reshape((7, 7, self.encoded_dims)),
            # Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu"),
            Conv2DTranspose(32, kernel_size=(3, 3), strides=(2, 2), padding="same", activation="relu"),
            # Conv2D(16, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu"),
            Conv2DTranspose(16, kernel_size=(3, 3), strides=(2, 2), padding="same", activation="relu"),
            Conv2D(1, kernel_size=(3, 3), strides=(1, 1), padding="same"),
            Flatten(),
            tfpl.IndependentBernoulli((28, 28, 1))
        ])

        negative_log_likelihood = lambda x, rv_x: -rv_x.log_prob(x)
        self.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss=negative_log_likelihood
        )

        try:
            self.load_weights(filepath=self.file_path)
            self.model_trained = True
        except:
            print("No predefined weights found")

    def call(self, inputs, **kwargs):
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        return decoded

    def fit(self, **kwargs):
        if not self.model_trained or self.retrain:
            super().fit(**kwargs)
            print("Storing learned weights...")
            self.save_weights(self.file_path)

    def get_config(self):
        pass
