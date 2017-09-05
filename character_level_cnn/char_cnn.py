#!/opt/anaconda2/bin/python
# -*- coding:utf-8 -*-

import numpy as np

from keras.layers import BatchNormalization, Concatenate, Conv2D, Dense, Dropout, Embedding, Flatten, Input, MaxPooling2D, Reshape
from keras.models import Model
from keras.optimizers import Adam


COMMENT_MAX_LEN = 300


def create_charcnn_model(embed_size=128, max_length=300, filter_sizes=(2, 3, 4, 5), filter_num=64):
    inp = Input(shape=(max_length,))
    emb = Embedding(0xffff, embed_size)(inp)
    emb_ex = Reshape((max_length, embed_size, 1))(emb)
    convs = []
    for filter_size in filter_sizes:
        conv = Conv2D(filter_num, (filter_size, embed_size), activation="relu")(emb_ex)
        pool = MaxPooling2D(pool_size=(max_length - filter_size + 1, 1))(conv)
        convs.append(pool)
    convs_merged = Concatenate()(convs)
    reshape = Flatten()(convs_merged)
    fc1 = Dense(64, activation="relu")(reshape)
    bn1 = BatchNormalization()(fc1)
    do1 = Dropout(0.6)(bn1)
    fc2 = Dense(1, activation='sigmoid')(do1)
    model = Model(inputs=inp, outputs=fc2)
    return model


def train(values, batch_size=100, epoch_count=100, max_length=300, model_filepath="model.h5"):
    model = create_charcnn_model(max_length=max_length)
    optimizer = Adam()
    model.compile(loss='binary_crossentropy',
                       optimizer=optimizer,
                       metrics=['accuracy'])
    # train
    input_values = []
    target_values = []
    for target_value, input_value in values:
        input_values.append(input_value)
        target_values.append(target_value)
    model.fit(np.array(input_values), np.array(target_values), epochs=epoch_count, batch_size=batch_size, verbose=1, validation_split=0.2)
    model.save(model_filepath)


def text2value(text, max_length=300):
    text_value = [ord(x) for x in text.strip().decode("utf-8")]
    text_value = texts_value[:max_length]
    if len(text_value) < max_length:
        text_value += ([0] * (max_length - len(text_value)))
    return text_value


if __name__ == "__main__":
    import random
    import time
    random.seed(time.time())

    values = []
    with open("texts.tsv") as f:
        for l in f:
            target, text = l.split("\t", 1)
            target = int(target)
            text = text2value(text, max_lenght=COMMENT_MAX_LEN)
            values.append((target, text))
    random.shuffle(values)
    train(values, batch_size=100, epoch_count=30, max_length=COMMENT_MAX_LEN)
