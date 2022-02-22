from keras.models import Sequential
from keras.layers import Dense, ReLU, Conv2D, Conv2DTranspose, MaxPooling2D, Dropout, Flatten


def init_model():
    model = Sequential()

    # TODO: find a good architecture
    # encoder
    # input: images of size 28 x 28 x 3
    # reduce dimensions, e.g. to 4

    # TODO: how to use transposed 2d convolutions
    # TODO: find out which scaling method to use
    # decoder
    # input: depends on dimensionality reduction of encoder
    # output: 28 x 28 x 3 images

    return model


class AutoEncoder:

    def __init__(self):
        # construct model using verification net as a template
        # use convolutions with stride > 1 for encoder
        # use transposed convolutions for the decoder
        # construct single net, i.e. input -> dim reduction (encoding) -> dim inflation (decoding)
        self.model = init_model()

    def train(self):
        # binary reconstruction loss for decoder
        pass

    def predict(self):
        # pass the given input through the encoder
        # pass the output of the encoder through the decoder
        # return the prediction from the decoder
        pass
