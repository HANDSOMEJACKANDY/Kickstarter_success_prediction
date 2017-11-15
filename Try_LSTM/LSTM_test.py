# LSTM for sequence classification in the IMDB dataset
import numpy as np
import pandas as pd
import gensim
import re
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import GRU
from keras.layers import Dropout
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.layers import BatchNormalization
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.utils import np_utils

# fix random seed for reproducibility
np.random.seed(7)

def is_desired_letter(char):
    return ord(char) >= 97 and ord(char) < 123 or ord(char) >= 48 and ord(char) < 58 or ord(char) == ord(".") or ord(char) == ord(",") or ord(char) == ord(" ")


def get_train_data(train_portion):
    # load the dataset but only keep the top n words, zero the rest
    train_data = pd.read_csv("input/kickstarter_train.csv")
    train_texts_and_results = train_data.iloc[:, [2, -1]]
    # do preliminary preprocessing:remove all symbols
    train_data["desc"] = [[char for char in str(text).lower() if is_desired_letter(char)] for
                          text in train_data["desc"]]
    # remove too short desc
    drop_index = []
    for i in range(len(train_data)):
        if len(train_data.iloc[i, 2]) <= 40:
            drop_index.append(i)
    train_data.drop(train_data.index[drop_index])
    # get final train data
    split_point = int(train_portion * len(train_data))
    train_texts = np.array(train_data.iloc[:split_point, 2])
    train_results = np.array(train_data.iloc[:split_point, -1])
    test_texts = np.array(train_data.iloc[split_point:, 2])
    test_results = np.array(train_data.iloc[split_point:, -1])
    return train_texts, train_results, test_texts, test_results


def convert_to_onehot(data, num_features):
    new_data = []
    for item in data:
        new_data.append(np_utils.to_categorical(item, num_classes=num_features))
    return np.array(new_data)


# get training testing data from disk
train_data_portion = 0.8
trainX, trainY, testX, testY = get_train_data(train_data_portion)
print("data grabbed")

# convert char to int
all_letters = sorted(set([char for text in trainX for char in text]))
char_to_int = dict((c, i+1) for i, c in enumerate(all_letters))
trainX = [[char_to_int[char] for char in text] for text in trainX]
testX = [[char_to_int[char] for char in text] for text in testX]
print("preliminary tokenizing finished")

# truncate and pad input sequences
max_desc_length = 200
trainX = sequence.pad_sequences(list(trainX), maxlen=max_desc_length, truncating="post")
testX = sequence.pad_sequences(list(testX), maxlen=max_desc_length, truncating="post")
print("padding finished")

# get one hot representation dict for all letters
# num_features = len(all_letters) + 1
# trainX = convert_to_onehot(trainX, num_features)
# testX = convert_to_onehot(testX, num_features)
# print("one hot representation finished")

# reshape trainX to multi_timestep single feature
time_steps = max_desc_length
num_features = 1
trainY = np.array(trainY)
trainY = trainY.reshape((-1, time_steps, num_features))
trainX = np.array(trainX)
trainX = trainX.reshape((-1, time_steps, num_features))
print("reshaping data with shape {}".format(trainX.shape))

# generate model
model = Sequential()
model.add(LSTM(units=128, input_shape=(time_steps, num_features)))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(units=128, activation="sigmoid"))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(units=128, activation="sigmoid"))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=["accuracy"])
print("model building finished\n", model.summary())

# do training
model.fit(trainX, trainY, epochs=20, batch_size=128, validation_split=0.1, verbose=1)

# save model
filepath = "{epoch:02d}-{loss:.4f}.hdf5"
model.save(filepath)

# present result on test data
scores = model.evaluate(testX, testY, verbose=1)
print("Accuracy:{}".format(scores[1]*100))

