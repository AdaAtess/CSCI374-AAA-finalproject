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

"""
Given a model, save the pickled version 
Of it with name model_name.
Returns True if successful
"""
def storeModel(model, model_name):
    pickle.dump(model, open(model_name, 'wb'))
    return True

"""
Given a model name, open its pickled model and 
Return it
"""
def loadModel(model_name):
    model = pickle.load(open(model_name), 'rb')
    return model


tensorflow.random.set_seed(1)

"""
Main Program for YikYakYeo 
"""

def handle_abbreviation(data):
    replacers = {'dm': 'direct message',
                 'thx': 'thanks',
                 'dming': 'direct messaging',
                 'dmed': 'direct messaged',
                 'pls': 'please',
                 'plz': 'please',
                 'u': 'you',
                 'asap': 'as soon as possible',
                 '...': '',
                 '. . .': '',
                 'r': 'are',
                 'ur': 'your',
                 'atm': 'at the moment',
                 'brt': 'be right there',
                 'brb': 'be right back',
                 'imo': 'in my opinion',
                 'btw': 'by the way',
                 'cya': 'see you later',
                 'gr8': 'great',
                 'm8': 'mate',
                 'lol': 'laughing out loud',
                 'lmao': 'laughed my ass off',
                 'lmfao': 'laughed my fucking ass off',
                 'idk': 'I dont know',
                 'tbh': 'to be honest',
                 'tbf': 'to be fair',
                 'wtf': 'what the fuck',
                 'u2': 'you too',
                 'irl': 'in real life',
                 'idfk': 'I dont fucking know',
                 'idek': 'I dont even know',
                 'pov': 'point of view',
                 'b4': 'before',
                 'bff': 'best friends forever',
                 'i': 'I',
                 'bf': 'boyfriend',
                 'gf': 'girlfriend',
                 'cmon': 'come on',
                 'jfc': 'jesus fucking christ',
                 'imho': 'in my humble opinion',
                 'fyi': 'for your information',
                 'smh': 'shake my head',
                 'omw': 'on my way',
                 'wth': 'what the heck',
                 'sm': 'so much',
                 'ily': 'I love you',
                 'lysm': 'I love you so much',
                 'ts': 'this shit',
                 'ilym': 'I love you more',
                 'idc': 'I dont care',
                 'rly': 'really',
                 'rlly': 'really',
                 'idrm': 'I dont really mind',
                 'proly': 'probably',
                 'prolly': 'probably',
                 'prob': 'probably',
                 'probs': 'probably',
                 'omg': 'oh my god',
                 'cos': 'because',
                 'cuz': 'because',
                 'cus': 'because',
                 'iykyk': 'if you know you know',
                 'oml': 'oh my lord',
                 'so': 'significant other',
                 'wyd': 'what you doing',
                 'jk': 'just kidding',
                 'tw': 'trigger warning',
                 'cw': 'content warning',
                 'wya': 'where you at',
                 'wdym': 'what do you mean',
                 'gm': 'good morning',
                 'gn': 'good night',
                 'ppl': 'people',
                 'nvm': 'never mind',
                 'ily2': 'I love you too',
                 'atp': 'at this point',
                 'kys': 'kill yourself',
                 'kms': 'kill myself',
                 'hbd': 'happy birthday',
                 'hw': 'homework',
                 'gud': 'good',
                 'wud': 'what you doing',
                 'wywd': 'what you wanna do',
                 'ikr': 'I know right',
                 'k': 'okay',
                 'l8r': 'later',
                 'nbd': 'no big deal',
                 'np': 'no problem',
                 'yw': 'your welcome',
                 'sry': 'sorry',
                 'srs': 'serious',
                 'srsly': 'seriously',
                 'sup': 'whats up',
                 'str8': 'straight',
                 'tgif': 'thank god its friday',
                 'wbu': 'what about you',
                 'hbu': 'how about you',
                 'w/o': 'without',
                 'wrk': 'work',
                 'yolo': 'you only live once',
                 'i': 'I',
                 'b00bs': 'boobs',
                 'prospie': 'prospective student',
                 'bestie': 'best friend',
                 'ngl': 'not gonna lie',
                 'rn': 'right now',
                 'mf': 'motherfucker'}
    data.replace(replacers, regex=True, inplace=True)
    return data

def standardize_text(text):
    txt = "".join(v for v in text if v not in string.punctuation).lower()
    return txt


# ========= KAGGLE CODE =========


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


# ========= end: KAGGLE CODE =========


def main():
    # load data
    input_file = "yikyakyeo/data_sets/medium_yikyakyeo_100_count.csv"
    # input_file = "yikyakyeo/data_sets/yikyakyeo.csv"
    data = pd.read_csv(input_file)
    # handles abbreviations here
    data = handle_abbreviation(data)
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
    # TODO: will experiment in changing num epochs here
    # (verbose prints training progress, i.e. 'x/epochs')
    model.fit(predictors, label, epochs=50, verbose=2)

    # generate text
    seed_text = "my"  # can be anything (TODO: will change to set to maybe random)
    next_words = 5  # num of next words to predict following seed_text, TODO: experiment with
    print(generate_text(seed_text, next_words, model, max_sequence_len, tokenizer))


if __name__ == '__main__':
    main()
