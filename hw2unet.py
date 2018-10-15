import numpy as np 
import pandas as pd
from keras.preprocessing import text, sequence
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from sklearn.model_selection import train_test_split
from keras.metrics import categorical_accuracy
from keras import backend as K
import tensorflow as tf
from keras import optimizers
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import KFold


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



# train_df = pd.read_csv('train.csv')
train_df = pd.read_csv('training_data.csv')
test_df = pd.read_csv('test.csv')

maxlen_seq = 2048

# Loading and converting the inputs to trigrams
train_input_seqs, train_target_seqs = train_df[['input', 'expected']][(train_df.len <= maxlen_seq)].values.T
train_input_grams = seq2ngrams(train_input_seqs)

# Same for test
test_input_seqs = test_df['input'].values.T
test_input_grams = seq2ngrams(test_input_seqs)

# Initializing and defining the tokenizer encoders and decoders based on the train set
tokenizer_encoder = Tokenizer()
tokenizer_encoder.fit_on_texts(train_input_grams)
tokenizer_decoder = Tokenizer(char_level = True)
tokenizer_decoder.fit_on_texts(train_target_seqs)

# Using the tokenizer to encode and decode the sequences for use in training
# Inputs
train_input_data = tokenizer_encoder.texts_to_sequences(train_input_grams)
train_input_data = sequence.pad_sequences(train_input_data, maxlen = maxlen_seq, padding = 'post')

# Targets
train_target_data = tokenizer_decoder.texts_to_sequences(train_target_seqs)
train_target_data = sequence.pad_sequences(train_target_data, maxlen = maxlen_seq, padding = 'post')
train_target_data = to_categorical(train_target_data)

# Use the same tokenizer defined on train for tokenization of test
test_input_data = tokenizer_encoder.texts_to_sequences(test_input_grams)
test_input_data = sequence.pad_sequences(test_input_data, maxlen = maxlen_seq, padding = 'post')

# Computing the number of words and number of tags to be passed as parameters to the keras model
n_words = len(tokenizer_encoder.word_index) + 1
n_tags = len(tokenizer_decoder.word_index) + 1

def get_model():
    input = Input(shape = (maxlen_seq,))

    # Defining an embedding layer mapping from the words (n_words) to a vector of len 128
    x = Embedding(input_dim = n_words, output_dim = 128, input_length = maxlen_seq)(input)
    conv1 = Conv1D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x)
    conv1 = Conv1D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    conv1 = BatchNormalization()(conv1)
    conv1 = Dropout(0.4)(conv1)
    pool1 = MaxPooling1D(pool_size=(2))(conv1)

    conv2 = Conv1D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv1D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = Dropout(0.4)(conv2)
    pool2 = MaxPooling1D(pool_size=(2))(conv2)

    conv3 = Conv1D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv1D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = Dropout(0.4)(conv3)
    pool3 = MaxPooling1D(pool_size=(2))(conv3)


    conv4 = Conv1D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv1D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    conv4 = BatchNormalization()(conv4)
    drop4 = Dropout(0.4)(conv4)
    pool4 = MaxPooling1D(pool_size=(2))(drop4)

    conv5 = Conv1D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv1D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    conv5 = BatchNormalization()(conv5)
    drop5 = Dropout(0.4)(conv5)

    up6 = Conv1D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling1D(size = (2))(drop5))
    up6 = BatchNormalization()(up6)
    merge6 = concatenate([drop4,up6], axis = 2)
    merge6 = Dropout(0.3)(merge6)
    conv6 = Conv1D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Conv1D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
    conv6 = BatchNormalization()(conv6)

    up7 = Conv1D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling1D(size = (2))(conv6))
    up7 = BatchNormalization()(up7)
    merge7 = concatenate([conv3,up7], axis = 2)
    merge7 = Dropout(0.3)(merge7)
    conv7 = Conv1D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Conv1D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
    conv7 = BatchNormalization()(conv7)

    up8 = Conv1D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling1D(size = (2))(conv7))
    up8 = BatchNormalization()(up8)
    merge8 = concatenate([conv2,up8], axis = 2)
    merge8 = Dropout(0.3)(merge8)
    conv8 = Conv1D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Conv1D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
    conv8 = BatchNormalization()(conv8)

    up9 = Conv1D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling1D(size = (2))(conv8))
    up9 = BatchNormalization()(up9)
    merge9 = concatenate([conv1,up9], axis = 2)
    merge9 = Dropout(0.3)(merge9)
    conv9 = Conv1D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Conv1D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = BatchNormalization()(conv9)

    concat = Dense(32, activation = 'relu')(conv9)
    y = Dense(n_tags, activation = 'softmax')(concat)


    # Defining the model as a whole and printing the summary
    model = Model(input, y)
    model.summary()

    adam = optimizers.Adam(lr=0.01)
    # Setting up the model with categorical x-entropy loss and the custom accuracy function as accuracy
    model.compile(optimizer = adam, loss = "categorical_crossentropy", metrics = ["accuracy", accuracy])
    
    return model

cross_val = True
if not cross_val:
    model = get_model()
    # Splitting the data for train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(train_input_data, train_target_data, test_size = .1, random_state = 0)

    filepath="weights.best.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]

    # Training the model on the training data and validating using the validation set
    model.fit(X_train, y_train, batch_size = 64, epochs = 20, validation_data = (X_val, y_val), callbacks=callbacks_list, verbose = 1)

    # Defining the decoders so that we can
    revsere_decoder_index = {value:key for key,value in tokenizer_decoder.word_index.items()}
    revsere_encoder_index = {value:key for key,value in tokenizer_encoder.word_index.items()}

    model.load_weights("weights.best.hdf5")
    y_test_pred = model.predict(test_input_data[:])
    result = []
    print(len(test_input_data))
    for i in range(len(test_input_data)):
        result.append(print_results(test_input_seqs[i], y_test_pred[i], revsere_decoder_index))

    df = pd.DataFrame(data={'id':test_df['id'], 'expected':result})
    df.to_csv('prediction.csv', index=False)

else:
    def get_callbacks(name_weights, patience_lr):
        mcp_save = ModelCheckpoint(name_weights, save_best_only=True, monitor='val_loss', mode='min', verbose=1)
        # reduce_lr_loss = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=patience_lr, verbose=1, epsilon=1e-4, mode='min')
        return [mcp_save]
    k=10
    cv_acc = []
    cv_loss = []
    folds = list(KFold(n_splits=k).split(train_input_data, train_target_data))
    for j, (train_idx, val_idx) in enumerate(folds):
        print('\nFold ',j)
        X_train_cv = train_input_data[train_idx]
        y_train_cv = train_target_data[train_idx]
        X_valid_cv = train_input_data[val_idx]
        y_valid_cv= train_target_data[val_idx]
        
        name_weights = "final_model_fold" + str(j) + "_weights.h5"
        callbacks = get_callbacks(name_weights = name_weights, patience_lr=10)
        model = get_model()
        model.fit(X_train_cv, y_train_cv,
                    batch_size = 128,
                    epochs=20,
                    shuffle=True,
                    verbose=1,
                    validation_data = (X_valid_cv, y_valid_cv),
                    callbacks = callbacks)
        model.load_weights(name_weights)
        y_val_pred = model.predict(X_valid_cv[:])
        result = []
        print(len(y_val_pred))
        for i in range(len(y_val_pred)):
            result.append(print_results(X_valid_cv[i], y_val_pred[i], revsere_decoder_index))

        df = pd.DataFrame(data={'id':val_idx, 'expected':result})
        df.to_csv('jz2979_jz2997_fold0{}.csv'.format(j+1), index=False)