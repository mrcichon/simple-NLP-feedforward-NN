import tensorflow as tf
import pandas as pd
import numpy as np

Tag = [1, 0]

df = pd.read_csv("data.csv")
df_train = df[df['dataset'].str.contains("train")]
x_train = [word.split() for word in df_train["Transaction descriptor"]]
y_train = [word for word in df_train["store_number"]]
df_y = pd.DataFrame(y_train)
df_x = pd.DataFrame(x_train).replace('#','',regex=True)
df_x = df_x.replace('-','',regex=True)
df_x.columns = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
df_x = df_x['A'].append(df_x['B']).append(df_x['C']).append(df_x['D']).\
    append(df_x['E']).append(df_x['F']).append(df_x['G'])
df_x = pd.DataFrame(df_x)
df_x = df_x.replace(to_replace='None', value=np.nan).dropna().reset_index()
df_x = df_x.drop(labels="index", axis=1)
df_x.columns = ["WORD"]

characters = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwyz '
def my_onehot_encoded(data):
    # define a mapping of chars to integers
    char_to_int = dict((c, i) for i, c in enumerate(characters))
    integer_encoded = [char_to_int[char] for char in data]
    onehot_encoded = list()
    for value in integer_encoded:
        character = [0 for _ in range(len(characters))]
        character[value] = 1
        onehot_encoded.append(character)

    return onehot_encoded


def my_onehot_decoded(onehot_encoded):
    int_to_char = dict((i, c) for i, c in enumerate(characters))
    words = list()
    for word in onehot_encoded:
        words_single = list()
        for char in word:
            inverted = int_to_char[np.argmax(char)]
            words_single.append(inverted)
        words.append(words_single)
    return words


def pad_data(data):
    pad_list2d = [0] * len(characters)
    data_len = [len(word) for word in data]
    for word in data:
        if len(word) < max(data_len):
            value = max(data_len) - len(word)
            for _ in range(value):
                word.append(pad_list2d)
    data_padded = data
    return data_padded


df_x["TAG"] = Tag[1]
for word in df_y.values:
    word = word[0]
    df_x.loc[df_x['WORD'] == word, 'TAG'] = Tag[0]


y = [tag for tag in df_x["TAG"]]
df_x = [word for word in df_x["WORD"]]

df_x = list(map(my_onehot_encoded, df_x))
df_x = pad_data(df_x)
df_x = np.asarray(df_x)
y = np.asarray(y)


model = tf.keras.Sequential(
    [

        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ]
)

model.compile(
    loss='binary_crossentropy', optimizer="adam", metrics=['accuracy']
)

model.fit(df_x, y, batch_size=1, epochs=8)
model.summary()

test = ["MCDONALDS", '26824', "483280353", "UT044",  "1003300", "STATEN"]
test = list(map(my_onehot_encoded, test))
test = pad_data(test)
test = np.asarray(test)

pred_letter = [letter for word_pred in model.predict(test) for letter in word_pred]
letter = [letter for word in my_onehot_decoded(test) for letter in word]
print(list(zip(pred_letter, letter)))
