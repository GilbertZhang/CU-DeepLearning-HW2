import numpy as np 
import pandas as pd
from keras.preprocessing import text, sequence
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Bidirectional
from sklearn.model_selection import train_test_split
from keras.metrics import categorical_accuracy
from keras import backend as K
import tensorflow as tf


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
def seq2ngrams(seqs, n = 3):
    return np.array([[seq[i : i + n] for i in range(len(seq))] for seq in seqs])



train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

maxlen_seq = 512

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

input = Input(shape = (maxlen_seq,))

# Defining an embedding layer mapping from the words (n_words) to a vector of len 128
x = Embedding(input_dim = n_words, output_dim = 128, input_length = maxlen_seq)(input)

# Defining a bidirectional LSTM using the embedded representation of the inputs
x = Bidirectional(LSTM(units = 256, return_sequences = True, recurrent_dropout = 0.1))(x)

# A dense layer to output from the LSTM's64 units to the appropriate number of tags to be fed into the decoder
y = TimeDistributed(Dense(n_tags, activation = "softmax"))(x)

# Defining the model as a whole and printing the summary
model = Model(input, y)
model.summary()
"""
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 512)               0         
_________________________________________________________________
embedding_1 (Embedding)      (None, 512, 128)          1225984   
_________________________________________________________________
bidirectional_1 (Bidirection (None, 512, 128)          98816     
_________________________________________________________________
time_distributed_1 (TimeDist (None, 512, 9)            1161      
=================================================================
Total params: 1,325,961
Trainable params: 1,325,961
Non-trainable params: 0

"""

# Setting up the model with categorical x-entropy loss and the custom accuracy function as accuracy
model.compile(optimizer = "rmsprop", loss = "categorical_crossentropy", metrics = ["accuracy", accuracy])

# Splitting the data for train and validation sets
X_train, X_val, y_train, y_val = train_test_split(train_input_data, train_target_data, test_size = .1, random_state = 0)

# Training the model on the training data and validating using the validation set
model.fit(X_train, y_train, batch_size = 128, epochs = 10, validation_data = (X_val, y_val), verbose = 1)

# Defining the decoders so that we can
revsere_decoder_index = {value:key for key,value in tokenizer_decoder.word_index.items()}
revsere_encoder_index = {value:key for key,value in tokenizer_encoder.word_index.items()}

y_test_pred = model.predict(test_input_data[:])
result = []
print(len(test_input_data))
for i in range(len(test_input_data)):
    result.append(print_results(test_input_seqs[i], y_test_pred[i], revsere_decoder_index))

df = pd.DataFrame(data={'id':test_df['id'], 'expected':result})
df.to_csv('prediction.csv', index=False)
