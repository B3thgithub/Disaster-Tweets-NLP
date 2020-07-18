import csv
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import numpy as np
import re
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import pandas as pd

# File paths 
root = "../input/nlp-getting-started/"
train = root + "train.csv"
test = root + "test.csv"

# Getting english stopwords
stop_words = stopwords.words('english')

def pre_process(file_name):
    num_tweets = 0
    labels = []
    tweets = []
    with open(file_name, 'r') as csv_file:
        data = csv.DictReader(csv_file)
        punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
        for row in data:
            sentence = row['text']
            # Remove punctuation
            sentence_no_punct = ""
            for char in sentence:
                if char not in punctuations:
                    sentence_no_punct = sentence_no_punct + char
            # Removing stopwords 
            sentence_nostop = ' '.join([word.lower() for word in  sentence_no_punct.split() if word.lower() not in stop_words])
            # Removing links
            sentence_nostop = re.sub(r"https?[A-Za-z0-9]+", "", sentence_nostop)
            tweets.append(str(sentence_nostop))
            if 'target' in row:
                labels.append(int(row['target']))
            num_tweets += 1
    return num_tweets, labels, tweets
           
num_train_tweets, train_labels, train_tweets = pre_process(train)
num_test_tweets, test_labels, test_tweets = pre_process(test)

# Creating validation set
split = 0.9
train_split_num = int(num_train_tweets*split)
valid_tweets = train_tweets[train_split_num:]
valid_labels = train_labels[train_split_num:]
train_tweets = train_tweets[:train_split_num]
train_labels = train_labels[:train_split_num]

# Tokenize words in training corpus
tokenizer = Tokenizer(num_words=10000, oov_token = "<OOV>", lower=True)
tokenizer.fit_on_texts(train_tweets)
word_index = tokenizer.word_index

def length_counter(tweets_list):
    """
    Returns the maximum sequence length of sentences in the list
    """
    longest_list = 0
    for sentence in tweets_list:
        sentence_length = len(sentence.split())
        if sentence_length > longest_list:
            longest_list = sentence_length
    return longest_list

max_train_len = length_counter(train_tweets)
max_valid_len = length_counter(valid_tweets)
max_test_len = length_counter(test_tweets)

max_sequence_len = max([max_train_len, max_valid_len, max_test_len])

# Set embedding parameters
vocab_size=10000
embedding_dim = 16
max_length = max_sequence_len
trunc_type = 'post'
padding_type = 'post'

# Make tokenized sequences of train data sentences 
train_sequences = tokenizer.texts_to_sequences(train_tweets)
train_padded = pad_sequences(train_sequences, padding=padding_type, maxlen=max_length, truncating=trunc_type)

valid_sequences = tokenizer.texts_to_sequences(valid_tweets)
valid_padded = pad_sequences(valid_sequences, padding=padding_type, maxlen=max_length, truncating=trunc_type)

test_sequences = tokenizer.texts_to_sequences(test_tweets)
test_padded = pad_sequences(test_sequences, padding=padding_type, maxlen=max_length, truncating=trunc_type)

# Convert labels to NP arrays
train_labels = np.array(train_labels)
valid_labels = np.array(valid_labels)

# Define Model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(6, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile Model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

# Train Model
EPOCHS = 10
history = model.fit(train_padded, train_labels, epochs=EPOCHS, validation_data=(valid_padded, valid_labels), verbose=1)

# Metrics
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
prec = history.history['precision']
val_prec = history.history['val_precision']
recall = history.history['recall']
val_recall = history.history['val_recall']

f1_scores = []
f1_valid_scores = []
for i in range(len(prec)):
    f1_scores.append((2*prec[i]*recall[i])/(prec[i]+recall[i]))
    f1_valid_scores.append((2*val_prec[i]*val_recall[i])/(val_prec[i]+val_recall[i]))

# Plotting accuracy
plt.figure(figsize=(18, 6))
plt.subplot(1, 3, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy', fontsize=12)
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch', fontsize=12)

# Plotting loss
plt.subplot(1, 3, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Loss', fontsize=12)
plt.title('Training and Validation Loss')
plt.xlabel('Epoch', fontsize=12)

# Plotting F1
plt.subplot(1, 3, 3)
plt.plot(f1_scores, label='Training F1 score')
plt.plot(f1_valid_scores, label='Validation F1 score')
plt.legend(loc='upper right')
plt.ylabel('F1', fontsize=12)
plt.title('Training and Validation F1 scores')
plt.xlabel('Epoch', fontsize=12)
plt.show()

# Inference
predictions = model.predict(test_padded)
submission = pd.read_csv(root+'sample_submission.csv')
submission.target = predictions
submission.target = submission.target.round(0).astype(int)
submission.to_csv('submission.csv',index=False)
