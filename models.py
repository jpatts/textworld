import tensorflow as tf
import numpy as np


class Seq2seq(tf.keras.Model):
    def __init__(self, vocab_size, embedding_size):
        super(Seq2seq, self).__init__()
        self.enc_hidden = tf.zeros((1, embedding_size))
        self.dec_hidden = tf.zeros((1, embedding_size))

        self.encoder = Encoder(vocab_size, embedding_size)
        self.decoder = Decoder(vocab_size, embedding_size)
    
    def call(self, x_in):
        encoded, self.enc_hidden = self.encoder(x_in, self.enc_hidden)
        x_out, self.dec_hidden = self.decoder(encoded, self.dec_hidden)
        return x_out


class Encoder(tf.keras.layers.Layer):
    def __init__(self, vocab_size, embedding_size):
        super(Encoder, self).__init__()
        
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_size)
        self.gru = tf.keras.layers.GRU(embedding_size)

    def call(self, x_in, hidden):
        embedded = self.embedding(x_in)
        encoded, hidden = self.gru(embedded, hidden)
        return encoded, hidden


class Decoder(tf.keras.layers.Layer):
    def __init__(self, vocab_size, embedding_size):
        super(Decoder, self).__init__()
        
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_size)
        self.relu = tf.keras.layers.ReLU()
        self.gru = tf.keras.layers.GRU(embedding_size)
        self.out = tf.keras.layers.Dense(vocab_size)

    def call(self, encoded, hidden):
        embedded = self.embedding(encoded)
        decoded, hidden = self.gru(self.relu(embedded), hidden)
        x_out = self.out(decoded)
        return x_out, hidden
