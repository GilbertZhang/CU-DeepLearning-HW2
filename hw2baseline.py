import numpy as np 
import pandas as pd
from keras.preprocessing import text, sequence
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.models import Model, Input
from keras.layers import *
from sklearn.model_selection import train_test_split
from keras.metrics import categorical_accuracy
from keras import backend as K
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from keras import optimizers
from sklearn.model_selection import KFold
import keras


def levenshtein(s1, s2):
    if len(s1) < len(s2):
        return levenshtein(s2, s1)

    # len(s1) >= len(s2)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1 # j+1 instead of j since previous_row and current_row are one character longer
            deletions = current_row[j] + 1       # than s2
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


# The custom accuracy metric used for this task
def accuracy(y_true, y_pred):
    y = tf.argmax(y_true, axis =- 1)
    y_ = tf.argmax(y_pred, axis =- 1)
    mask = tf.greater(y, 0)
    return K.cast(K.equal(tf.boolean_mask(y, mask), tf.boolean_mask(y_, mask)), K.floatx())

# Maps the sequence to a one-hot encoding
def onehot_to_seq(oh_seq, index):
    s = ''
    for o in oh_seq:
        i = np.argmax(o)
        if i != 0:
            s += index[i]
        else:
            break
    return s

# prints the results
def print_results(x, y_, revsere_decoder_index):
    # print("input     : " + str(x))
    # print("prediction: " + str(onehot_to_seq(y_, revsere_decoder_index).upper()))
    return str(onehot_to_seq(y_, revsere_decoder_index).upper())

# Computes and returns the n-grams of a particualr sequence, defaults to trigrams
def seq2ngrams(seqs, n = 1):
    return np.array([[seq[i : i + n] for i in range(len(seq))] for seq in seqs])



train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

maxlen_seq = 512

def preprocessing_tain_x(train_input_seqs, n):
    train_input_grams = seq2ngrams(train_input_seqs, n)
    tokenizer_encoder = Tokenizer()
    tokenizer_encoder.fit_on_texts(train_input_grams)
    # Using the tokenizer to encode and decode the sequences for use in training
    # Inputs
    train_input_data = tokenizer_encoder.texts_to_sequences(train_input_grams)
    train_input_data = sequence.pad_sequences(train_input_data, maxlen = maxlen_seq, padding = 'post')
    return train_input_data, tokenizer_encoder

def preprocessing_test_x(test_input_seqs, tokenizer_encoder, n):
    test_input_grams = seq2ngrams(test_input_seqs, n)
    # Use the same tokenizer defined on train for tokenization of test
    test_input_data = tokenizer_encoder.texts_to_sequences(test_input_grams)
    test_input_data = sequence.pad_sequences(test_input_data, maxlen = maxlen_seq, padding = 'post')
    return test_input_data


def preprocessing_y(train_target_seqs):
    # Initializing and defining the tokenizer encoders and decoders based on the train set
    tokenizer_decoder = Tokenizer(char_level = True)
    tokenizer_decoder.fit_on_texts(train_target_seqs)
    # Targets
    train_target_data = tokenizer_decoder.texts_to_sequences(train_target_seqs)
    train_target_data_ori = sequence.pad_sequences(train_target_data, maxlen = maxlen_seq, padding = 'post')
    train_target_data = to_categorical(train_target_data_ori)
    return train_target_data, tokenizer_decoder

# Loading
train_input_seqs, train_target_seqs = train_df[['input', 'expected']][(train_df.len <= maxlen_seq)].values.T
test_input_seqs = test_df['input'].values.T

train_input_data1, tokenizer_encoder1 = preprocessing_tain_x(train_input_seqs, 2)
test_input_data1 = preprocessing_test_x(test_input_seqs, tokenizer_encoder1, 2)
train_target_data, tokenizer_decoder = preprocessing_y(train_target_seqs)


# Computing the number of words and number of tags to be passed as parameters to the keras model
n_words1 = len(tokenizer_encoder1.word_index) + 1
n_tags = len(tokenizer_decoder.word_index) + 1

def get_model():
    input1 = Input(shape = (None,))

    # Defining an embedding layer mapping from the words (n_words) to a vector of len 128
    x1 = Embedding(input_dim = n_words1, output_dim = 128, input_length = None)(input1)

    x1 = Bidirectional(LSTM(units = 64, return_sequences = True, recurrent_dropout = 0))(x1)
    x1 = Bidirectional(LSTM(units = 64, return_sequences = True, recurrent_dropout = 0))(x1)
    x1 = Dense(32, activation="relu")(x1)
    # A dense layer to output from the LSTM's64 units to the appropriate number of tags to be fed into the decoder
    y = Dense(n_tags, activation = "softmax")(x1)

    # Defining the model as a whole and printing the summary
    model = Model(input1, y)
    model.summary()

    rmsprop = optimizers.Adam(lr=0.02)

    # Setting up the model with categorical x-entropy loss and the custom accuracy function as accuracy
    model.compile(optimizer = rmsprop, loss = "categorical_crossentropy", metrics = [accuracy])
    
    return model

cross_val = False
if not cross_val:
    
    model = get_model()
    filepath="weights.best.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]


    # Splitting the data for train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(train_input_data1, train_target_data, test_size = .1, random_state = 0)

    # Training the model on the training data and validating using the validation set
    model.fit(X_train, y_train, batch_size = 128, epochs = 20, validation_data = (X_val, y_val), callbacks=callbacks_list, verbose = 1)

    # Defining the decoders so that we can
    revsere_decoder_index = {value:key for key,value in tokenizer_decoder.word_index.items()}
    revsere_encoder_index = {value:key for key,value in tokenizer_encoder1.word_index.items()}

    model.load_weights("weights.best.hdf5")

    y_train_pred = model.predict(train_input_data1[:500])
    edit_dis = []
    for i in range(500):
        output = print_results(train_input_seqs[i], y_train_pred[i], revsere_decoder_index)
        edit_dis.append(levenshtein(output, train_input_seqs[i]))
    print(np.mean(edit_dis))

    y_test_pred = model.predict(test_input_data1[:])
    result = []
    
    for i in range(len(test_input_data1)):
        output = print_results(test_input_seqs[i], y_test_pred[i], revsere_decoder_index)
        result.append(output)
    df = pd.DataFrame(data={'id':test_df['id'], 'expected':result})
    df.to_csv('prediction.csv', index=False)

else:

    def get_callbacks(name_weights, patience_lr):
        mcp_save = ModelCheckpoint(name_weights, save_best_only=True, monitor='val_loss', mode='min')
        # reduce_lr_loss = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=patience_lr, verbose=1, epsilon=1e-4, mode='min')
        return [mcp_save]
    k=10
    cv_acc = []
    cv_loss = []
    folds = list(KFold(n_splits=k, shuffle=True, random_state=1).split(train_input_data1, train_target_data))
    for j, (train_idx, val_idx) in enumerate(folds):

        print('\nFold ',j)
        X_train_cv = train_input_data1[train_idx]
        y_train_cv = train_target_data[train_idx]
        X_valid_cv = train_input_data1[val_idx]
        y_valid_cv= train_target_data[val_idx]
        
        name_weights = "final_model_fold" + str(j) + "_weights.h5"
        callbacks = get_callbacks(name_weights = name_weights, patience_lr=10)
        model = get_model()
        model.fit(X_train_cv, y_train_cv,
                    batch_size = 128,
                    epochs=1,
                    shuffle=True,
                    verbose=1,
                    validation_data = (X_valid_cv, y_valid_cv),
                    callbacks = callbacks)
        
        loss, _, acc = model.evaluate(X_valid_cv, y_valid_cv)
        cv_acc.append(acc)
        cv_loss.append(loss)
    print("Average Accuracy: {}, Average Loss: {}".format(np.mean(cv_acc), np.mean(cv_loss)))