import tensorflow_datasets as tfds
import tensorflow as tf

from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from Transformer.TransformerEncoder import TransformerEncoder
from Transformer.TransformerDecoder import TransformerDecoder

from Transformer.Transformer_Loss_Optimizer import Get_Custom_Adam_Optimizer, Transformer_Loss_AIAYN

BUFFER_SIZE = 20000
BATCH_SIZE = 64
MAX_LENGTH = 40

tf.enable_eager_execution()
conf = tf.ConfigProto()
conf.gpu_options.allow_growth= True
sess = tf.Session(config=conf)
tf.keras.backend.set_session(sess)

def Get_Transformer_Model(transformer_encoder_layers, transformer_decoder_layers, model_depth, ff_depth, num_heads, SOURCE_SIZE, TARGET_SIZE,
                          POS_ENC_INPUT, POS_ENC_TARGET,
                          MAX_SEQ_LEN_INPUT, MAX_SEQ_LEN_TARGET):
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

    output = Dense(TARGET_SIZE)(transformer_decoder)

    transformer_optimizer = Get_Custom_Adam_Optimizer(model_depth)

    model = Model([input_encoder, input_decoder, encoder_padding_mask, look_ahead_mask, decoder_padding_mask], output)
    model.compile(optimizer= transformer_optimizer, loss= Transformer_Loss_AIAYN)
    model.summary()
    return model

def create_padding_mask(seq, paddingValue):
    seq = tf.cast(tf.math.equal(seq, paddingValue), tf.float32)
    returnedseq = seq[:, tf.newaxis, tf.newaxis, :]
    # add extra dimensions to add the padding
    # to the attention logits.
    return returnedseq # (batch_size, 1, 1, seq_len)

def create_look_ahead_mask(size):
  mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
  return mask  # (seq_len, seq_len)

def create_masks(inp, tar, paddingvalueinp, paddingvaluetar):
    # Encoder padding mask
    enc_padding_mask = create_padding_mask(inp, paddingvalueinp)

    # Used in the 2nd attention block in the decoder.
    # This padding mask is used to mask the encoder outputs.
    dec_padding_mask = create_padding_mask(inp, paddingvalueinp)

    # Used in the 1st attention block in the decoder.
    # It is used to pad and mask future tokens in the input received by
    # the decoder.
    look_ahead_mask = create_look_ahead_mask(tar.shape[1])
    dec_target_padding_mask = create_padding_mask(tar, paddingvaluetar)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    return enc_padding_mask.numpy(), combined_mask.numpy(), dec_padding_mask.numpy()

def encode(lang1, lang2):
    lang1 = [tokenizer_pt.vocab_size] + tokenizer_pt.encode(
        lang1.numpy()) + [tokenizer_pt.vocab_size + 1]

    lang2 = [tokenizer_en.vocab_size] + tokenizer_en.encode(
        lang2.numpy()) + [tokenizer_en.vocab_size + 1]

    return lang1, lang2

def tf_encode(pt, en):
  result_pt, result_en = tf.py_function(encode, [pt, en], [tf.int64, tf.int64])
  result_pt.set_shape([None])
  result_en.set_shape([None])

  return result_pt, result_en

def filter_max_length(x, y, max_length=MAX_LENGTH):
  return tf.logical_and(tf.size(x) <= max_length,
                        tf.size(y) <= max_length)

def train_step(model, input, target):
    tar_inp = target[:, :-1]
    tar_real = target[:, 1:]
    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(input, tar_inp, 0,0)

    hist = model.fit(x=[input, tar_inp, enc_padding_mask, combined_mask, dec_padding_mask], y=tar_real, epochs=1, verbose=0)
    return hist.history['loss'][0]


if __name__ == '__main__':
    examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en', with_info=True, as_supervised=True)
    train_examples, val_examples = examples['train'], examples['validation']
    tokenizer_en = tfds.features.text.SubwordTextEncoder.build_from_corpus(
        (en.numpy() for pt, en in train_examples), target_vocab_size=2 ** 13)

    tokenizer_pt = tfds.features.text.SubwordTextEncoder.build_from_corpus(
        (pt.numpy() for pt, en in train_examples), target_vocab_size=2 ** 13)

    train_dataset = train_examples.map(tf_encode)
    train_dataset = train_dataset.filter(filter_max_length)
    # cache the dataset to memory to get a speedup while reading from it.
    train_dataset = train_dataset.cache()
    train_dataset = train_dataset.shuffle(BUFFER_SIZE).padded_batch(BATCH_SIZE, padded_shapes=(BATCH_SIZE, MAX_LENGTH))
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    val_dataset = val_examples.map(tf_encode)
    val_dataset = val_dataset.filter(filter_max_length).padded_batch(BATCH_SIZE, padded_shapes=(BATCH_SIZE, MAX_LENGTH))

    input_vocab_size = tokenizer_pt.vocab_size + 2
    target_vocab_size = tokenizer_en.vocab_size + 2

    model = Get_Transformer_Model(transformer_encoder_layers=6,transformer_decoder_layers=6, model_depth=512, ff_depth=2048,
                                  num_heads=8, SOURCE_SIZE=input_vocab_size, TARGET_SIZE=target_vocab_size,
                                  POS_ENC_INPUT=input_vocab_size, POS_ENC_TARGET= target_vocab_size,
                                  MAX_SEQ_LEN_INPUT=64,
                                  MAX_SEQ_LEN_TARGET= MAX_LENGTH-1)

    for EPOCH in range(20):
        loss = 0
        batchnum = 0
        for(batch, (inp, tar)) in enumerate(train_dataset):
            loss_step = train_step(model, inp, tar)
            loss += loss_step
            batchnum += 1

        print(f'| Epoch {EPOCH} | Loss: {str(loss/batchnum)} | Batches: {batchnum}')



