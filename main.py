# keras module for building LSTM
from keras.utils import pad_sequences
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping
from keras.models import Sequential
import keras.utils as ku
import pandas as pd
import numpy as np
import string, os

# set seeds for reproducibility
import tensorflow
from numpy.random import seed

import pickle

# ========= PICKLE FILE CODE =========
"""
Store data in pickle file
Returns True if successful
"""
def storeData(filename, data):
    dbfile = open(filename, 'ab')
    pickle.dump(data, dbfile)
    dbfile.close()
    return True
  
def loadData(filename):
    dbfile = open(filename, 'rb')
    db = pickle.load(dbfile)
    dbfile.close()
    return db

tensorflow.random.set_seed(1)

"""
Main Program for YikYakYeo 
"""


# ========= KAGGLE CODE =========
def standardize_text(text):
    txt = "".join(v for v in text if v not in string.punctuation).lower()
    return txt


def get_sequence_of_tokens(yaks, tokenizer):
    # tokenization
    tokenizer.fit_on_texts(yaks)
    total_words = len(tokenizer.word_index) + 1

    # convert data to sequence of tokens
    input_sequences = []
    for line in yaks:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i + 1]
            input_sequences.append(n_gram_sequence)
    return input_sequences, total_words


def generate_padded_sequences(input_sequences, total_words):
    max_sequence_len = max([len(x) for x in input_sequences])
    input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

    predictors, label = input_sequences[:, :-1], input_sequences[:, -1]
    label = ku.to_categorical(label, num_classes=total_words)
    return predictors, label, max_sequence_len


# TODO: experiment in changing stuff here
def create_model(max_sequence_len, total_words):
    input_len = max_sequence_len - 1
    model = Sequential()

    # Add Input Embedding Layer
    model.add(Embedding(total_words, 10, input_length=input_len))

    # Add Hidden Layer 1 - LSTM Layer
    model.add(LSTM(100))
    model.add(Dropout(0.1))

    # Add Output Layer
    model.add(Dense(total_words, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam')

    return model


def generate_text(seed_text, next_words, model, max_sequence_len, tokenizer):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
        predicted = np.argmax(model.predict(token_list), axis=-1)

        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " " + output_word
    return seed_text


# ========= KAGGLE CODE =========


def main():
    # load data
    input_file = "yikyakyeo/data_sets/medium_yikyakyeo.csv"
    # input_file = "yikyakyeo/data_sets/yikyakyeo.csv"
    data = pd.read_csv(input_file)
    # creates array for all yaks from 'text' column
    yaks = data['text'].tolist()
    for i in range(len(yaks)):
        yaks[i] = standardize_text(yaks[i])

    # generate sequence of N-gram Tokens for the model to predict the next token
    # (every int in input_sequences corresponds to the index of a word in the whole vocabulary)
    tokenizer = Tokenizer()
    input_sequences, total_words = get_sequence_of_tokens(yaks, tokenizer)
    # pad sequences so that they're of same length
    predictors, label, max_sequence_len = generate_padded_sequences(input_sequences, total_words)

    # LSTM model
    model = create_model(max_sequence_len, total_words)
    # model.summary()
    # train
    # TODO: will experiment in changing stuff here
    model.fit(predictors, label, epochs=50, verbose=5)

    # generate text
    seed_text = "my"  # can be anything (TODO: will change to set to maybe random)
    next_words = 5  # num of next words to predict
    print(generate_text(seed_text, next_words, model, max_sequence_len, tokenizer))


if __name__ == '__main__':
    main()
