import tensorflow as tf
import numpy as np


class Seq2seq(tf.keras.Model):
    def __init__(self, vocab_size, embedding_size, batch_size, units, start, end):
        super(Seq2seq, self).__init__()
        self.encoder = Encoder(vocab_size, embedding_size, batch_size, units)
        self.decoder = Decoder(vocab_size, embedding_size, units)
        self.batch_size = batch_size
        self.start = start
        self.end = end
    
    def call(self, enc_in, max_entity_length):
        encoded, self.enc_hidden = self.encoder(enc_in)
        dec_in = tf.expand_dims([self.start] * self.batch_size, 1)
        predictions = tf.zeros((self.batch_size, max_entity_length), dtype=tf.dtypes.int64)
        dec_hidden = self.enc_hidden
        for t in range(2 + (2*max_entity_length)):
            x_out, self.dec_hidden, attn = self.decoder(dec_in, dec_hidden, encoded)
            pred = tf.reshape(tf.argmax(x_out, 1), [self.batch_size, 1])
            predictions = tf.concat([predictions[:, :t], pred], 1)
            # end tokens generated
            if tf.reduce_all(tf.equal(pred, self.end)):
                break
            dec_in = pred
        
        return predictions


class Encoder(tf.keras.layers.Layer):
    def __init__(self, vocab_size, embedding_size, batch_size, enc_units):
        super(Encoder, self).__init__()
        self.enc_hidden = tf.zeros((batch_size, enc_units))
        
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_size)
        self.gru = tf.keras.layers.GRU(enc_units, return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform')

    def call(self, x_in):
        # x_in shape = (batch_size, num_entites * max_entity_length)
        embedded = self.embedding(x_in)
        encoded, self.enc_hidden = self.gru(embedded, initial_state=self.enc_hidden)
        # (batch_size, num_entities * max_entity_length, units), (batch_size, units)
        return encoded, self.enc_hidden


class BahdanauAttention(tf.keras.Model):
  def __init__(self, units):
    super(BahdanauAttention, self).__init__()
    self.W1 = tf.keras.layers.Dense(units)
    self.W2 = tf.keras.layers.Dense(units)
    self.V = tf.keras.layers.Dense(1)

  def call(self, enc_hidden, encoded):
    # (batch_size, 1, hidden size)
    hidden_with_time_axis = tf.expand_dims(enc_hidden, 1)

    # (batch_size, max_length, 1)
    score = self.V(tf.nn.tanh(
        self.W1(encoded) + self.W2(hidden_with_time_axis)))

    # (batch_size, max_length, 1)
    attention_weights = tf.nn.softmax(score, axis=1)

    context_vector = attention_weights * encoded
    # (batch_size, hidden_size)
    context_vector = tf.reduce_sum(context_vector, axis=1)

    return context_vector, attention_weights
    

class Decoder(tf.keras.layers.Layer):
    def __init__(self, vocab_size, embedding_size, dec_units):
        super(Decoder, self).__init__()
        self.dec_units = dec_units
        
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_size)
        self.gru = tf.keras.layers.GRU(dec_units, return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform')
        self.fc = tf.keras.layers.Dense(vocab_size)
        self.attention = BahdanauAttention(self.dec_units)

    def call(self, x_in, enc_hidden, encoded):
        # (batch_size, max_length, hidden_size)
        context_vector, attention_weights = self.attention(enc_hidden, encoded)

        # (batch_size, 1, embedding_dim)
        embedded = self.embedding(x_in)

        # (batch_size, 1, embedding_dim + hidden_size)
        concatenated = tf.concat([tf.expand_dims(context_vector, 1), embedded], axis=-1)

        decoded, dec_hidden = self.gru(concatenated)
        
        # (batch_size * 1, hidden_size)
        decoded = tf.reshape(decoded, (-1, decoded.shape[2]))

        # (batch_size, vocab)
        x_out = self.fc(decoded)

        return x_out, dec_hidden, attention_weights
