import tensorflow as tf

from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from Transformer.TransformerEncoder import TransformerEncoder
from Transformer.TransformerDecoder import TransformerDecoder

from Transformer.Transformer_Loss_Optimizer import Get_Custom_Adam_Optimizer, Transformer_Loss_AIAYN
import numpy as np

import sys

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

    output = Dense(TARGET_SIZE, activation='softmax')(transformer_decoder)

    transformer_optimizer = Get_Custom_Adam_Optimizer(model_depth)

    model = Model([input_encoder, input_decoder, encoder_padding_mask, look_ahead_mask, decoder_padding_mask], output)
    model.compile(optimizer= transformer_optimizer, loss= 'categorical_crossentropy')
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

def LoadData(dataLoc, dataFile, type, samples):
    Y = []
    YSequence = []
    loadedSamples = 0
    with open(dataFile) as paths:
        line = paths.readline()
        while line:
            encodingPath = dataLoc + line.split("\t")[type]
            encodingPath = encodingPath.split("\n")[0]
            yfile = open(encodingPath)
            # KRN and SKM files. Process it as a block.
            if (type == 3):
                krnLines = yfile.readlines()
                for i, line in enumerate(krnLines):
                    line = line.split("\n")[0]  # Dumb trick to get the characters without breaks
                    krnLines[i] = line
                YSequence = krnLines
            else:
                YSequence = yfile.readline().split("\t")  # Load the agnostic file

            Y.append(YSequence)
            YSequence = []
            yfile.close()
            line = paths.readline()

            loadedSamples += 1
            if loadedSamples == samples:
                break

    return np.array(Y)


def make_vocabulary(YSequences, pathToSave, nameOfVoc):
    vocabulary = set()
    stride = 1

    for samples in YSequences:
        for sequence in samples:
            vocabulary.update(sequence)

    # Vocabulary created
    w2i = {symbol: idx+stride for idx, symbol in enumerate(vocabulary)}
    i2w = {idx+stride: symbol for idx, symbol in enumerate(vocabulary)}

    w2i['<pad>'] = 0
    i2w[0] = '<pad>'

    # Save the vocabulary
    np.save(pathToSave + "/" + nameOfVoc + "w2i.npy", w2i)
    np.save(pathToSave + "/" + nameOfVoc + "i2w.npy", i2w)

    return w2i, i2w


def batch_confection(batchX, batchY, max_len_input, max_len_target, targetLength, w2iagnostic, w2ikern):
    encoder_input = np.full((len(batchX), max_len_input), w2iagnostic['<pad>'], dtype=np.float)
    decoder_input = np.full((len(batchY), max_len_target), w2iskm['<pad>'], dtype=np.float)
    decoder_output = np.zeros((len(batchY), max_len_target, targetLength), dtype=np.float)

    for i, sequence in enumerate(batchX):
        for j, char in enumerate(sequence):
            encoder_input[i][j] = w2iagnostic[char]

    for i, sequence in enumerate(batchY):
        for j, char in enumerate(sequence):

            decoder_input[i][j] = w2ikern[char]

            if j > 0:
                decoder_output[i][j - 1][w2ikern[char]] = 1.

    return encoder_input, decoder_input, decoder_output


def batch_generator(X, Y, batch_size, sourceLength, maxlengthinput, maxlengthtarget, targetLength, w2iagnostic, w2itarget):
    index = 0
    while True:
        BatchX = X[index:index + batch_size]
        BatchY = Y[index:index + batch_size]

        encoder_input, decoder_input, decoder_output = batch_confection(BatchX, BatchY, maxlengthinput, maxlengthtarget,
                                                                                targetLength, w2iagnostic, w2itarget)

        #We have to generate THE MASKS
        input_padding_mask, combined_mask, target_padding_mask = create_masks(encoder_input, decoder_input, w2iagnostic['<pad>'], w2iskm['<pad>'])

        yield [encoder_input, decoder_input, input_padding_mask, combined_mask, target_padding_mask], decoder_output

        index = (index + batch_size) % len(X)

def edit_distance(a, b):
    n, m = len(a), len(b)

    if n > m:
        a, b = b, a
        n, m = m, n

    current = range(n + 1)
    for i in range(1, m + 1):
        previous, current = current, [i] + [0] * n
        for j in range(1, n + 1):
            add, delete = previous[j] + 1, current[j - 1] + 1
            change = previous[j - 1]
            if a[j - 1] != b[i - 1]:
                change = change + 1
            current[j] = min(add, delete, change)

    return current[n]

def validateModel(model, testGen, i2wskm):
    current_edition_val = 0
    inputs, ground_truth = next(testGen)
    predictions = model.predict(inputs, batch_size=200)
    for i, prediction in enumerate(predictions):
        raw_sequence = [i2wskm[char] for char in np.argmax(prediction, axis=1)]
        raw_trueseq = [i2wskm[char] for char in np.argmax(ground_truth[i], axis=1)]
        predictionSequence = []
        truesequence = []

        for char in raw_sequence:
            predictionSequence += [char]
            if char == '<eos>':
                break
        for char in raw_trueseq:
            truesequence += [char]
            if char == '<eos>':
                break

        if i == 0:
            print("Prediction: " + str(predictionSequence))
            print("True: " + str(truesequence))

        current_edition_val += edit_distance(truesequence, predictionSequence) / len(truesequence)

    return current_edition_val

if __name__ == '__main__':
    X = LoadData("",
                 "Dataset/dataset.lst", 1, 10200)
    Y = LoadData("",
                 "Dataset/dataset.lst", 3, 10200)

    X = [['<sos>'] + sequence + ['<eos>'] for sequence in X]
    Y = [['<sos>'] + sequence + ['<eos>'] for sequence in Y]
    maxLengthInput = len(X[np.argmax([len(element) for element in X])])
    maxLengthTarget = len(Y[np.argmax([len(element) for element in Y])])

    XTest = X[10000:]
    YTest = Y[10000:]

    X = X[:10000]
    Y = Y[:10000]

    print(X[0])
    print(Y[0])

    w2iagnostic, i2wagnostic = make_vocabulary([X, XTest], "vocabulary", "agnostic")
    w2iskm, i2wskm = make_vocabulary([Y, YTest], "vocabulary", "skm")

    batch_gen = batch_generator(X, Y, 8, len(w2iagnostic), maxLengthInput, maxLengthTarget, len(w2iskm), w2iagnostic, w2iskm)

    model  = Get_Transformer_Model(transformer_encoder_layers=6,
                                   transformer_decoder_layers=6,
                                   model_depth=512,
                                   ff_depth=2048,
                                   num_heads=8,
                                   SOURCE_SIZE=len(w2iagnostic),
                                   TARGET_SIZE=len(w2iskm),
                                   POS_ENC_INPUT=len(w2iagnostic),
                                   POS_ENC_TARGET=len(w2iskm),
                                   MAX_SEQ_LEN_INPUT=maxLengthInput,
                                   MAX_SEQ_LEN_TARGET=maxLengthTarget)

    for SUPER_EPOCH in range(20):
        model.fit_generator(batch_gen, steps_per_epoch=len(X)//8, epochs=1)
        test_generator = batch_generator(XTest, YTest, 200, len(w2iagnostic), maxLengthInput, maxLengthTarget, len(w2iskm), w2iagnostic, w2iskm)
        edition_val = validateModel(model, test_generator, i2wskm)
        SER = (100. * edition_val) / len(XTest)
        print(f'| Epoch {(SUPER_EPOCH + 1) * 5} | SER in validation with cheating: {SER}')



