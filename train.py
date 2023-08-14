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
alpha_size = 26
max_word_length = 15
model = Sequential()
model = tf.keras.models.load_model("model.h5")
# model.add(LSTM(128, input_shape=(max_word_length, alpha_size+1), return_sequences=True))
# # model.add(Dense(128, input_shape=(max_word_length, alpha_size+1), activation='relu'))
# # model.add(Dense(128, activation='relu'))
# model.add(LSTM(128, return_sequences=True))
# model.add(Flatten())
# model.add(Dense(1920, activation='relu'))
# # model.add(Dense(128, activation='relu'))
# model.add(Dense(alpha_size, activation='softmax'))
# model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
print(model.summary())
max_blanks = 9
words_lod = np.loadtxt('words_250000_train.txt', dtype=str)
char_to_int = {char: i for i, char in enumerate('abcdefghijklmnopqrstuvwxyz_')}
int_to_char = {i: char for i, char in enumerate('abcdefghijklmnopqrstuvwxyz_')}

def create_input(seq, max_word_length=15, vocab_sie=27):
    ans = []
    for i in seq:
        buf = [0]*vocab_size
        buf[char_to_int[i]] = 1
        ans.append(buf)
    while(len(ans) < max_word_length):
        ans.append([0]*vocab_sie)
    return ans


for kl in range(1000):
    lens = np.array([len(w) for w in words_lod])
    # print('99th percentile of word lengths: {}'.format(percentile_99))
    words_lod = words_lod[lens <= max_word_length]
    vocab_size = len(char_to_int)
    test_words, val_words = train_test_split(words_lod, test_size=0.1, random_state=random.randint(0, 100000))
    words = test_words
    # print('Number of words after filtering: {}'.format(len(words)))
    output_letters = []
    input_words = []
    from collections import defaultdict
    for cur, word in enumerate(words[:100000]):
        input_seq = ["_"]*(len(word))
        seter = set(word)
        inputer = create_input(input_seq)
        target = random.choice(list(seter))
        input_words.append(inputer)
        output_letters.append(char_to_int[target])
        seter.remove(target)
        while(len(seter)):
            for i in range(len(word)):
                if word[i] == target:
                    input_seq[i] = target
            inputer = create_input(input_seq)
            target = random.choice(list(seter))
            input_words.append(inputer)
            output_letters.append(char_to_int[target])
            seter.remove(target)
    print(len(input_words))
    input_words = np.array(input_words)
    output_letters = np.array(output_letters)
    output_letters = tf.keras.utils.to_categorical(output_letters, num_classes=alpha_size)
    print(output_letters.shape)

    # print(len(input_words))
    # input_words = np.array(input_words)
    # output_letters = np.array(output_letters)
    # output_letters = tf.keras.utils.to_categorical(output_letters, num_classes=alpha_size)
    # print(output_letters.shape)
    # padded = tf.keras.preprocessing.sequence.pad_sequences(input_words, maxlen=max_word_length, padding='post')
    # print(padded)
    # print(output_letters)
    model.fit(input_words, output_letters, epochs=1, verbose=1)
    # break
    model.save('model.h5')
print('Training complete')
# print(test_words[0])
