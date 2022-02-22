
class AutoEncoder:

    def __init__(self):
        # construct model using verification net as a template
        # use convolutions with stride > 1 for encoder
        # use transposed convolutions for the decoder
        # construct two nets: encoder, decoder
        pass

    def train(self):
        # binary reconstruction loss for decoder
        pass

    def predict(self):
        # pass the given input through the encoder
        # pass the output of the encoder through the decoder
        # return the prediction from the decoder
        pass
