#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  6 18:07:15 2018

@author: jai
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from wordcloud import WordCloud
import collections
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
import tensorflow as tf

data=pd.read_csv('reviews.csv')

#EDA
data.head(5)

plt.hist(data['Division Name'])

plt.hist(data['Department Name'])

plt.scatter(data['Clothing ID'], data['Rating'])
plt.xlabel('Clothing ID')
plt.ylabel('Rating')

sns.boxplot(x='Rating', y='Age', data=data)

plt.scatter(data['Positive Feedback Count'], data['Rating'])
plt.xlabel('Positive Feedback Count')
plt.ylabel('Rating')

review_length = data['Review Text'].astype(str).apply(len)

plt.hist(data['Age'])

ax = sns.boxplot(data['Age'], review_length)
ax.set_xticklabels(ax.get_xticklabels(), rotation=75, ha="right")
ax.set(ylabel="Review Text Length")
#plt.scatter(data['Age'], review_length, color='red', alpha=0.05)

plt.hist(data['Age'], bins=200, alpha=0.75)

ax = sns.pointplot(data['Positive Feedback Count'], data['Rating'], color='purple')
ax.set_xticklabels(ax.get_xticklabels(), rotation=75, ha="right")
plt.scatter(data['Positive Feedback Count'], data['Rating'])


#Word Corpus

text = ""

titles = data['Title']
titles = titles.dropna()

for word in titles:
    text = text + str(word) + " "

wordcloud = WordCloud().generate(text)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")


text = ""
reviews = data['Review Text']
reviews = reviews.dropna()

for word in reviews:
    text = text + str(word)

wordcloud = WordCloud().generate(text)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")


#Preprocessing

#train = train.drop(train[train.Rating == 3].index)
#train['Good'] = train['Rating'] >= 4

data.dropna(subset=['Review Text', 'Division Name'], inplace=True)
train, test = train_test_split(data, test_size=0.2)


words = []
for word in train['Review Text']:
    words += (str(word).lower()).split()    

dictionary = collections.Counter(words)

list_to_remove = list(dictionary)
for item in list_to_remove:
    if item.isalpha() == False: 
        del dictionary[item]
    elif len(item) == 1:
        del dictionary[item]
        
dictionary = dictionary.most_common(3000)


tokenizer = RegexpTokenizer(r'\w+')
train['Tokenized'] = train.apply(lambda row: tokenizer.tokenize(str(row['Review Text'])), axis=1) 

list = []

for word in dictionary:
    list += word
    
word_list = [x for x in list if not isinstance(x, int)]

final_dictionary = collections.Counter(word_list)

i=0
for item in final_dictionary:
    final_dictionary[item] = i
    i+=1

def encode_tokens(tokens):
    encoded_tokens = tokens[:]
    for i, token in enumerate(tokens):
        if token.lower() in final_dictionary:
            encoded_tokens[i] = final_dictionary[token]
    return encoded_tokens


print(encode_tokens(train['Tokenized'][20427]))

train['Encoded'] = train.apply(lambda row: encode_tokens(row['Tokenized']), axis=1) 

for row in train['Encoded']:
    for word in row:
        if type(word)==int:
            pass
        else:
            row.remove(word)
        
        
#RNN Model

X_train = []
train.apply(lambda row: X_train.append(row['Encoded'][:-1]), axis=1)

Y_train = []
train.apply(lambda row: Y_train.append(row['Encoded'][1:]), axis=1)

word_dim=3000
hidden_dim=100
bptt_truncate=4

U = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (hidden_dim, word_dim))
V = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (word_dim, hidden_dim))
W = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (hidden_dim, hidden_dim))

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def forward_propagation(x):
    # The total number of time steps
    T = len(x)
    # During forward propagation we save all hidden states in s because need them later.
    # We add one additional element for the initial hidden, which we set to 0
    s = np.zeros((T + 1, hidden_dim))
    s[-1] = np.zeros(hidden_dim)
    # The outputs at each time step. Again, we save them for later.
    o = np.zeros((T, word_dim))
    # For each time step...
    for t in np.arange(T):
        # Note that we are indxing U by x[t]. This is the same as multiplying U with a one-hot vector.
        s[t] = np.tanh(U[:,x[t]] + W.dot(s[t-1]))
        o[t] = softmax(V.dot(s[t]))
    return [o, s]

def predict(x):
    # Perform forward propagation and return index of the highest score
    o, s = forward_propagation(x)
    return np.argmax(o, axis=1)

np.random.seed(10)
o, s = forward_propagation(X_train[10])
print(o.shape)
print(o)


def calculate_total_loss(x, y):
    L = 0
    # For each sentence...
    for i in np.arange(len(y)):
        o, s = forward_propagation(x[i])
        # We only care about our prediction of the "correct" words
        correct_word_predictions = o[np.arange(len(y[i])), y[i]]
        # Add to the loss based on how off we were
        L += -1 * np.sum(np.log(correct_word_predictions))
    return L
 
def calculate_loss(x, y):
    # Divide the total loss by the number of training examples
    N = np.sum((len(y_i) for y_i in y))
    return calculate_total_loss(x,y)/N

print("Expected Loss for random predictions: %f" % np.log(3000))
print("Actual loss: %f" % calculate_loss(X_train[:1000], Y_train[:1000]))




def generate_sentence():
    i=0
    # We start the sentence with the start token
    new_sentence = [0]
    # Repeat until we get an end token
    while(i<7):
        i+=1
        next_word_probs = forward_propagation(new_sentence)
        #print(next_word_probs[:-1])
        #samples = np.random.multinomial(1, next_word_probs[:-1])
        sampled_word = np.argmax(next_word_probs[:-1])
        new_sentence.append(sampled_word)
    sentence_str = decode_tokens(new_sentence)
    return sentence_str
 
num_sentences = 10
sentence_min_length = 7
 
for i in range(num_sentences):
    sentence = []
    # We want long sentences, not sentences with one or two words
    while len(sent)<sentence_min_length:
        sentence = generate_sentence()
    print(" ".join(sentence))



index_to_word = [x[0] for x in final_dictionary.keys]


def decode_tokens(sentence):
    decoded_tokens = sentence[:]
    for i, word in enumerate(sentence):
        if word in final_dictionary:
            decoded_tokens[i] = final_dictionary[word]
    return decoded_tokens

#Keras Implementation
from keras import Sequential    

num_steps=30

model = Sequential()
model.add(Embedding(word_dim, hidden_dim, input_length=num_steps))
model.add(LSTM(hidden_dim, return_sequences=True))
model.add(LSTM(hidden_dim, return_sequences=True))
if use_dropout:
    model.add(Dropout(0.5))
model.add(TimeDistributed(Dense(vocabulary)))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
