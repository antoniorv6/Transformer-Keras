import tensorflow as tf
from tensorflow.keras.layers import Embedding, Dropout
import numpy as np
from Transformer.DecoderLayer import DecoderLayer

class TransformerDecoder(tf.keras.layers.Layer):

    @staticmethod
    def get_angles(pos, i, model_depth):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(model_depth))
        return pos * angle_rates

    @staticmethod
    def positional_encoding(position, model_depth):
        angle_rads = TransformerDecoder.get_angles(np.arange(position)[:, np.newaxis],
                                                   np.arange(model_depth)[np.newaxis, :],
                                                   model_depth)
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        pos_encoding = angle_rads[np.newaxis, ...]

        return tf.cast(pos_encoding, dtype=tf.float32)

    def __init__(self, num_layers, model_depth, num_heads, feed_forward_depth, target_vocab_size, maximum_position_encoding, dropout_rate=0.1):
        super(TransformerDecoder, self).__init__()
        self.model_depth = model_depth
        self.num_layers = num_layers

        self.embedding = Embedding(target_vocab_size, model_depth)
        self.position_encoding = TransformerDecoder.positional_encoding(maximum_position_encoding, model_depth)

        self.decoder_layers = [DecoderLayer(model_depth, num_heads, feed_forward_depth, dropout_rate)
                               for _ in range(num_layers)]

        self.dropout = Dropout(dropout_rate)

    def call(self, inputs, encoder_output, look_ahead_mask, padding_mask):

        seq_len = tf.shape(inputs)[1]
        attention_weights = {}

        x = self.embedding(inputs) # (batch_size, target_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.model_depth, tf.float32))
        x += self.position_encoding[:, :seq_len, :]

        x = self.dropout(x)

        for i in range(self.num_layers):
            x, block1, block2 = self.decoder_layers[i](x,
                                                       encoder_output=encoder_output,
                                                       look_ahead_mask=look_ahead_mask,
                                                       padding_mask = padding_mask)
            attention_weights['decoder_layer{}_block1'.format(i+1)] = block1
            attention_weights['decoder_layer{}_block2'.format(i+1)] = block2

        return x, attention_weights