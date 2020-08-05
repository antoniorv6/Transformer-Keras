import tensorflow as tf
from Transformer import MHA

from Transformer.TransformerEncoder import TransformerEncoder
from Transformer.TransformerDecoder import TransformerDecoder

from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

from Transformer.Transformer_Loss_Optimizer import Get_Custom_Adam_Optimizer

import numpy as np

def initSession():
    conf = tf.ConfigProto()
    conf.gpu_options.allow_growth= True
    sess = tf.Session(config=conf)
    tf.keras.backend.set_session(sess)

def Get_Transformer_Model(transformer_encoder_layers, transformer_decoder_layers, model_depth, ff_depth, num_heads, SOURCE_SIZE, TARGET_SIZE,
                          POS_ENC_INPUT, POS_ENC_TARGET,
                          MAX_SEQ_LEN_INPUT, MAX_SEQ_LEN_TARGET):
    initSession()
    input_encoder = Input(shape=(None,))
    input_decoder = Input(shape=(None,))

    encoder_padding_mask = Input(shape=(None, 1, MAX_SEQ_LEN_INPUT))
    decoder_padding_mask = Input(shape=(None, 1, MAX_SEQ_LEN_INPUT))
    look_ahead_mask = Input(shape=(None, MAX_SEQ_LEN_TARGET, MAX_SEQ_LEN_TARGET))

    transformer_encoder = TransformerEncoder(num_layers=transformer_encoder_layers,
                                             model_depth= model_depth,
                                             num_heads= num_heads,
                                             feed_forward_depth= ff_depth,
                                             input_vocab_size= SOURCE_SIZE,
                                             maximum_pos_encoding= POS_ENC_INPUT
                                             )(input_encoder, mask=encoder_padding_mask)
    transformer_decoder, attn = TransformerDecoder(num_layers= transformer_decoder_layers,
                                             model_depth= model_depth,
                                             num_heads = num_heads,
                                             feed_forward_depth= ff_depth,
                                             target_vocab_size= TARGET_SIZE,
                                             maximum_position_encoding= POS_ENC_TARGET)(input_decoder,
                                                                                     encoder_output=transformer_encoder,
                                                                                     look_ahead_mask = look_ahead_mask,
                                                                                     padding_mask = decoder_padding_mask)

    output = Dense(TARGET_SIZE, activation='softmax')(transformer_decoder)

    transformer_optimizer = Get_Custom_Adam_Optimizer(model_depth=model_depth)

    model = Model([input_encoder, input_decoder, encoder_padding_mask, decoder_padding_mask, look_ahead_mask], output)
    model.compile(optimizer=transformer_optimizer, loss='categorical_crossentropy')
    model.summary()
    return model


def create_padding_mask(seq):
    seq = np.equal(seq, 0)

    # add extra dimensions to add the padding
    # to the attention logits.
    return seq[:, np.newaxis, np.newaxis, :]  # (batch_size, 1, 1, seq_len)

def create_look_ahead_mask(size):
  mask = 1 - np.triu(np.ones((size, size)), -1)
  return mask  # (seq_len, seq_len)


def create_masks(inp, tar):
    # Encoder padding mask
    enc_padding_mask = create_padding_mask(inp)

    # Used in the 2nd attention block in the decoder.
    # This padding mask is used to mask the encoder outputs.
    dec_padding_mask = create_padding_mask(inp)

    # Used in the 1st attention block in the decoder.
    # It is used to pad and mask future tokens in the input received by
    # the decoder.
    look_ahead_mask = create_look_ahead_mask(tar.shape[1])
    dec_target_padding_mask = create_padding_mask(tar)
    combined_mask = np.maximum(dec_target_padding_mask, look_ahead_mask)

    return enc_padding_mask, combined_mask, dec_padding_mask

if __name__ == '__main__':
    multiHeadAttention = MHA.MHA(model_depth=512, num_heads=8)
    y = tf.random.uniform((1,60,512)) #(batch_size, encoder_sequence, model_depth)
    out, attn = multiHeadAttention(y, k=y, q=y, mask=None)
    print(f'MHA output shape => {out.shape}')
    print(f'MHA attention shape => {attn.shape}')

    encoderLayer = EncoderLayer(model_depth=512, num_heads=8, feed_forward_depth=2048)
    output_layer_encoder = encoderLayer(tf.random.uniform((64,43,512)), mask=None)

    print(f'Encoder layer output shape => {output_layer_encoder.shape}')

    decoder_layer = DecoderLayer(512, 8, 2048)
    decoder_layer_output, _, _ = decoder_layer(tf.random.uniform((64,50,512)), encoder_output=output_layer_encoder, look_ahead_mask=None, padding_mask=None)
    print(f'Decoder layer output shape => {decoder_layer_output.shape}')

    encoder = TransformerEncoder(num_layers=2, model_depth=512, num_heads=8, feed_forward_depth=2048, input_vocab_size=8500, maximum_pos_encoding=10000)
    encoder_input = tf.random.uniform((64,62), dtype=tf.int64, minval=0, maxval=200)

    encoder_output = encoder(encoder_input, mask=None)

    print(f'Encoder output => {encoder_output.shape}')

    decoder = TransformerDecoder(num_layers=2, model_depth=512, num_heads=8, feed_forward_depth=2048, target_vocab_size=8000,
                                 maximum_position_encoding=5000)

    decoder_input = tf.random.uniform((64,26), dtype=tf.int64, minval=0, maxval=200)

    output, attn = decoder(decoder_input,
                           encoder_output=encoder_output,
                           look_ahead_mask = None,
                           padding_mask = None)

    print(f'Decoder output => {output.shape}')

    model = Get_Transformer_Model(2, 2, 512, 2048, 8, 8500, 8000, 10000, 6000, 38,36)

    temp_input = np.random.uniform(0,200, size=(64,38))
    temp_target = np.random.uniform(0,200, size=(64,36))

    input_padding_mask, combined_mask, target_padding_mask = create_masks(temp_input, temp_target)

    prediction = model.predict([temp_input, temp_target, input_padding_mask, target_padding_mask, combined_mask])

    print(f'Transformer prediction shape => {prediction.shape}')


