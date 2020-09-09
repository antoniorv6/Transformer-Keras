import tensorflow as tf
from tensorflow.keras.optimizers.schedules import LearningRateSchedule
from tensorflow.keras.optimizers import Adam

from .TransformerEncoder import TransformerEncoder
from .TransformerDecoder import TransformerDecoder
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction='none')

class TransformerLearningRate(LearningRateSchedule):
    def __init__(self, model_depth, warmup_steps=4000):
        super(TransformerLearningRate, self).__init__()

        self.model_depth = tf.cast(model_depth, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.model_depth) * tf.math.minimum(arg1, arg2)


def Get_Custom_Adam_Optimizer(model_depth):
    scheduler = TransformerLearningRate(model_depth)
    t_optimizer = Adam(scheduler, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
    return t_optimizer

def Transformer_Loss_AIAYN(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real,0))
    loss_ = loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_sum(loss_)/tf.reduce_sum(mask)

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

    output = Dense(TARGET_SIZE, activation='softmax')(transformer_decoder)

    transformer_optimizer = Get_Custom_Adam_Optimizer(model_depth)

    model = Model([input_encoder, input_decoder, encoder_padding_mask, look_ahead_mask, decoder_padding_mask], output)
    model.compile(optimizer= transformer_optimizer, loss= Transformer_Loss_AIAYN)
    model.summary()
    return model