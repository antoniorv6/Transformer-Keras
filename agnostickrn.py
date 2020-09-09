import tensorflow as tf
from Transformer.TransformerCore import Get_Transformer_Model
import numpy as np
import tqdm

tf.enable_eager_execution()
conf = tf.ConfigProto()
conf.gpu_options.allow_growth= True
sess = tf.Session(config=conf)
tf.keras.backend.set_session(sess)

BATCH_SIZE = 64

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
    decoder_output = np.zeros((len(batchY), max_len_target), dtype=np.float)

    for i, sequence in enumerate(batchX):
        for j, char in enumerate(sequence):
            encoder_input[i][j] = w2iagnostic[char]

    for i, sequence in enumerate(batchY):
        for j, char in enumerate(sequence):

            decoder_input[i][j] = w2ikern[char]

            if j > 0:
                decoder_output[i][j - 1] = w2ikern[char]

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

def validateModel(model, testGen, i2wskm, numOfBatches):
    current_edition_val = 0
    predictionSequence = []
    truesequence = []
    for _ in range(numOfBatches):
        inputs, ground_truth = next(testGen)
        predictions = model.predict(inputs, batch_size=BATCH_SIZE)
        for i, prediction in enumerate(predictions):
            raw_sequence = [i2wskm[char] for char in np.argmax(prediction, axis=1)]
            raw_trueseq = [i2wskm[char] for char in ground_truth[i]]
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

            current_edition_val += edit_distance(truesequence, predictionSequence) / len(truesequence)

    print("Prediction: " + str(predictionSequence))
    print("True: " + str(truesequence))

    return current_edition_val


def test_model(model, testGen, w2iagnostic, w2iskm, i2wskm, numOfBatches, max_inp_length, max_tar_len, target_len):
    current_edition_val = 0
    prediction = []
    gt = []
    for _ in range(numOfBatches):
        inputs, ground_truth = next(testGen)
        inputs = inputs[0]
        for i, inp in enumerate(inputs):
            prediction, gt = test_sequence(inp, ground_truth[i], w2iagnostic, w2iskm, i2wskm, model, max_tar_len)
            current_edition_val+=edit_distance(gt, prediction)

    print("Prediction: " + str(prediction))
    print("True: " + str(gt))

    return current_edition_val


def test_sequence(sequence, trueSequence, w2iagnostic, w2iskm, i2wskm, model, max_tar_len):
    decoded = np.full((1, max_tar_len), w2iskm['<pad>'], dtype=np.float)
    decoded[0][0] = w2iskm['<sos>']
    predicted = []
    true = []
    for character in trueSequence:
        if character == '<eos>':
            break
        true += [i2wskm[character]]

    for i in range(1, max_tar_len-1):
        input_padding_mask, combined_mask, target_padding_mask = create_masks([sequence], decoded,
                                                                              w2iagnostic['<pad>'], w2iskm['<pad>'])
        prediction = model.predict(x=[[sequence], decoded, input_padding_mask, combined_mask, target_padding_mask], steps=1, verbose=0)
        decoded[0][i] = np.argmax(prediction[0][i])

        if i2wskm[decoded[0][i]] == '<eos>':
            break

        predicted.append(i2wskm[decoded[0][i]])

    return predicted, true

if __name__ == '__main__':
    X = LoadData("",
                 "Dataset/dataset.lst", 1, 75000)
    Y = LoadData("",
                 "Dataset/dataset.lst", 3, 75000)

    X = [['<sos>'] + sequence + ['<eos>'] for sequence in X]
    Y = [['<sos>'] + sequence + ['<eos>'] for sequence in Y]
    maxLengthInput = len(X[np.argmax([len(element) for element in X])])
    maxLengthTarget = len(Y[np.argmax([len(element) for element in Y])])

    XTest = X[45000:]
    YTest = Y[45000:]

    X = X[:45000]
    Y = Y[:45000]

    XValidation = XTest[:1000]
    YValidation = YTest[:1000]


    print(X[0])
    print(Y[0])

    w2iagnostic, i2wagnostic = make_vocabulary([X, XTest], "vocabulary", "agnostic")
    w2iskm, i2wskm = make_vocabulary([Y, YTest], "vocabulary", "skm")

    batch_gen = batch_generator(X, Y, BATCH_SIZE, len(w2iagnostic), maxLengthInput, maxLengthTarget, len(w2iskm), w2iagnostic, w2iskm)

    model  = Get_Transformer_Model(transformer_encoder_layers=4,
                                   transformer_decoder_layers=4,
                                   model_depth=128,
                                   ff_depth=512,
                                   num_heads=8,
                                   SOURCE_SIZE=len(w2iagnostic),
                                   TARGET_SIZE=len(w2iskm),
                                   POS_ENC_INPUT=len(w2iagnostic),
                                   POS_ENC_TARGET=len(w2iskm),
                                   MAX_SEQ_LEN_INPUT=maxLengthInput,
                                   MAX_SEQ_LEN_TARGET=maxLengthTarget)

    for SUPER_EPOCH in range(3):
        model.fit_generator(batch_gen, steps_per_epoch=len(X)//BATCH_SIZE, epochs=5, verbose=2)
        val_generator = batch_generator(XTest, YTest, BATCH_SIZE, len(w2iagnostic), maxLengthInput, maxLengthTarget, len(w2iskm), w2iagnostic, w2iskm)
        test_generator = batch_generator(XValidation, YValidation, BATCH_SIZE, len(w2iagnostic), maxLengthInput, maxLengthTarget, len(w2iskm), w2iagnostic, w2iskm)
        edition_val = validateModel(model, val_generator, i2wskm, numOfBatches= len(XTest)//BATCH_SIZE)
        SERDECINP = (100. * edition_val) / len(XTest)
        edition_val_nodecinp = test_model(model=model, testGen=test_generator, w2iagnostic=w2iagnostic, w2iskm=w2iskm,
                                          i2wskm=i2wskm, numOfBatches=len(XValidation) // BATCH_SIZE,
                                          max_inp_length=maxLengthInput, max_tar_len=maxLengthTarget,
                                          target_len=len(w2iskm))

        SER = (100. * edition_val_nodecinp) / len(XTest)
        print(f'| Epoch {(SUPER_EPOCH + 1)} | Validation SER with input: {SERDECINP} | Real SER: {SER}')



