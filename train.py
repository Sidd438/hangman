import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, Flatten
from itertools import combinations
from sklearn.model_selection import train_test_split
import random

def generate_combinations(k, n):
    valid_combinations = []
    
    for comb in combinations(range(1, n + 1), k):
        valid_combinations.append(comb)
    
    return valid_combinations

words = np.loadtxt('words_250000_train.txt', dtype=str)
lens = np.array([len(w) for w in words])
percentile_99 = np.percentile(lens, 99)
# print('99th percentile of word lengths: {}'.format(percentile_99))
words = words[lens <= percentile_99]
char_to_int = {char: i for i, char in enumerate('abcdefghijklmnopqrstuvwxyz')}
vocab_size = len(char_to_int)
test_words, val_words = train_test_split(words, test_size=0.2, random_state=42)
words = test_words
# print('Number of words after filtering: {}'.format(len(words)))
max_blanks = 9
alpha_size = 26
max_word_length = 17
model = Sequential()
model.add(Dense(100, input_shape=(max_word_length,)))
model.add(Dense(100, activation='relu'))
model.add(Dense(26, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

output_letters = []
input_words = []
for cur, word in enumerate(words[:2500]):
    combs = []
    bls = random.sample(range(1, len(word)+1), min(4, len(word)))
    for i in bls:
        combs.extend(generate_combinations(i, len(word)))
    word_seq = [char_to_int[char] for char in word]
    print(word, word_seq, cur)
    # print(combs)
    for comb in combs:
        temp_seq = list(word_seq)+[26]*(max_word_length-len(word_seq))
        needs = []
        for index in comb:
            needs.append(temp_seq[index - 1])
            temp_seq[index - 1] = alpha_size
        for need in needs:
            input_words.append(temp_seq)
            output_letters.append(need)
input_words = np.array(input_words)
output_letters = np.array(output_letters)
output_letters = tf.keras.utils.to_categorical(output_letters, num_classes=alpha_size)
padded = tf.keras.preprocessing.sequence.pad_sequences(input_words, maxlen=max_word_length, padding='post')
# print(padded)
# print(output_letters)
model.fit(padded, output_letters, epochs=3, verbose=0)
    # break
print(output_letters)
print('Training complete')

model.save('model.h5')