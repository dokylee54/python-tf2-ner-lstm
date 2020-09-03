from tensorflow.keras.layers import Dense, Embedding, LSTM, Bidirectional, Dropout
from tensorflow.keras.models import Sequential, Model

import json


class MyModel(Model):
    def __init__(self, config):
        super(MyModel, self).__init__()
        self.embedding = Embedding(input_dim=config['vocab_size'], output_dim=config['embedding_dim'], input_length=config['maxlen'], mask_zero=True)
        self.dropout1 = Dropout(config['dropout_rate'])
        self.bilstm = Bidirectional(LSTM(32, return_sequences=True))
        self.dropout2 = Dropout(config['dropout_rate'])
        self.dense1 = Dense(config['dense_feature_dim'], activation='tanh')
        self.dropout3 = Dropout(config['dropout_rate'])
        self.dense2 = Dense(config['ner_tag_size'], activation='softmax')

    def call(self, x, training):
        x = self.embedding(x)
        x = self.dropout1(x, training=training)
        x = self.bilstm(x)
        x = self.dropout2(x, training=training)
        x = self.dense1(x)
        x = self.dropout3(x, training=training)
        return self.dense2(x)