from __future__ import print_function

from .DataExtractor import *
from .Seq2SeqModels import TeacherForcingModel, InferenceModel
from keras.layers import Activation
from keras.utils.generic_utils import get_custom_objects
from keras.callbacks import Callback
from keras import backend as K
import numpy as np


def ContractedSigmoid(x):
    return 1.0 / (1.0 + K.exp(-100.0 * x))


get_custom_objects().update({
    'contracted_sigmoid': Activation(ContractedSigmoid, name='contracted_sigmoid')})


class EarlyStoppingByAccuracy(Callback):
    def __init__(self, monitor='accuracy', value=0.75, verbose=0):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.value = value
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if current >= self.value:
            if self.verbose > 0:
                print("Epoch %05d: early stopping THR" % epoch)
            self.model.stop_training = True


class Text2TextModel:
    def __init__(self,
                 activation='sigmoid',
                 optimizer='adam',
                 loss='binary_crossentropy',
                 latent_dim=256,
                 metrics=['accuracy']):

        self.activation = activation
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        self.encoder_input_data = []
        self.decoder_input_data = []
        self.decoder_target_data = []
        self.train_model = None
        self.predict_model = None
        self.max_encoder_seq_length = 0
        self.max_decoder_seq_length = 0
        self.latent_dim = latent_dim

    def load_data(self, data_path, samples=10000):

        self.encoder_input_data, \
            self.decoder_input_data, \
            self.decoder_target_data, \
            self.max_encoder_seq_length, \
            self.max_decoder_seq_length = \
            VectorizeText2TextData(data_path, samples)

        self.train_model = TeacherForcingModel(
            self.latent_dim,
            self.activation,
            self.optimizer,
            self.loss,
            self.metrics
        )

    def train(self, batch_size=64, epochs=100, validation_split=0.2):

        # callback = EarlyStoppingByAccuracy()

        self.train_model.model.fit(
            [self.encoder_input_data, self.decoder_input_data],
            self.decoder_target_data,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=validation_split)

        self.predict_model = InferenceModel(
            self.train_model.model,
            self.latent_dim
        )

    def save(self, path):
        self.train_model.model.save(path)

    def predict(self, input_seq):
        print(ByteSecuenceToString(input_seq[0]))
        states_value = self.predict_model.encoder_model.predict(input_seq)
        target_seq = np.zeros((1, 1, 8))
        target_char = CharToBinaryArray('\t')
        target_seq[0, 0] = target_char
        stop_condition = False
        decoded_sentence = ""
        while not stop_condition:
            output_tokens, h, c = self.predict_model.decoder_model.predict([target_seq] + states_value)
            sampled_char = output_tokens[0, -1]
            decoded_char = BinaryArrayToChar(sampled_char)
            decoded_sentence += decoded_char
            if decoded_char == "\n" or len(decoded_sentence) > self.max_decoder_seq_length:
                stop_condition = True
            target_seq[0, 0] = sampled_char
            states_value = [h, c]
        return decoded_sentence
