import json
import string
import random
from datetime import datetime, date

from tkinter import *
from tkinter import ttk

import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
import keras
from keras import Sequential
from keras.layers import Dense, Dropout

# Downloads
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("wordnet")

# Load intents
with open('intents.json', 'r') as f:
    data = json.load(f)

words = []
classes = []
data_x = []
data_y = []

# Collect words and classes
for intent in data["intents"]:
    for pattern in intent["patterns"]:
        tokens = nltk.word_tokenize(pattern)
        words.extend(tokens)
        data_x.append(pattern)
        data_y.append(intent["tag"])

    if intent["tag"] not in classes:
        classes.append(intent["tag"])

lemmatizer = WordNetLemmatizer()

# FIXED: correct list comprehension
words = [
    lemmatizer.lemmatize(word.lower())
    for word in words
    if word not in string.punctuation
]

words = sorted(set(words))
classes = sorted(set(classes))

# Prepare training data
training = []
out_empty = [0] * len(classes)

for idx, doc in enumerate(data_x):
    bow = []
    text = nltk.word_tokenize(doc.lower())
    text = [lemmatizer.lemmatize(word) for word in text]

    for word in words:
        bow.append(1 if word in text else 0)
    
    output_row = list(out_empty)
    output_row[classes.index(data_y[idx])] = 1

    training.append([bow, output_row])

random.shuffle(training)
training = np.array(training, dtype=object)

train_x = np.array(list(training[:, 0]))
train_y = np.array(list(training[:, 1]))

# Build model
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation="softmax"))

adam = keras.optimizers.Adam(learning_rate=0.01)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=["accuracy"])

print(model.summary())
model.fit(x=train_x, y=train_y, epochs=150, verbose=1)

# --- Helper functions ---

def clean_text(text):
    tokens = nltk.word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return tokens

def bag_of_words(text, vocab):
    tokens = clean_text(text)
    bow = [0] * len(vocab)

    for w in tokens:
        for idx, word in enumerate(vocab):
            if word == w:
                bow[idx] = 1

    return np.array(bow)

def pred_class(text, vocab, labels):
    bow = bag_of_words(text, vocab)
    result = model.predict(np.array([bow]))[0]

    thresh = 0.5
    y_pred = [[idx, res] for idx, res in enumerate(result) if res > thresh]
    y_pred.sort(key=lambda x: x[1], reverse=True)

    return [labels[r[0]] for r in y_pred]

def get_response(intents_list, intents_json):
    if len(intents_list) == 0:
        return "Sorry! I don't understand."

    tag = intents_list[0]

    for intent in intents_json["intents"]:
        if intent["tag"] == tag:

            if tag == "time":
                return random.choice(intent["responses"]) + " " + str(datetime.now().time())

            if tag == "date":
                return random.choice(intent["responses"]) + " " + str(date.today())

            return random.choice(intent["responses"])

    return "Sorry! I don't understand."


# --- Chat loop ---
print("Press 0 if you don't want to chat with our Chatbot.")

while True:
    message = input("")
    if message == "0":
        break
    intents = pred_class(message, words, classes)
    result = get_response(intents, data)
    print(result)