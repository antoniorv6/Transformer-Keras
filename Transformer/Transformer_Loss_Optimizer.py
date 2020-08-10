import tensorflow as tf
from tensorflow.keras.optimizers.schedules import LearningRateSchedule
from tensorflow.keras.optimizers import Adam

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