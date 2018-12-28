# -*- coding: utf-8 -*-

import numpy as np
from sklearn.preprocessing import LabelEncoder
import keras as krs

BINARY_FLAG = "binary"
MULTI_FLAG = "multi"

def read_file(label, file_name):
    titles = []
    print("正在加载`{}`的数据...".format(label))
    with open(file_name, "r") as f:
        for line in f.readlines():
            if not line or not line.strip():
                continue
            titles.append(line.strip())
    return titles

def load_data_label():
    file_label_path = [
        ('健康类', 'data/health.txt'),
        ('科技类', 'data/tech.txt'),
        ('设计类', 'data/design.txt'),
    ]

    array_list = []
    all_titles = []
    for index, (label, file_name) in enumerate(file_label_path):
        titles = read_file(label, file_name)
        all_titles.extend(titles)
        arr = np.array([index]).repeat(len(titles))  # [index] * len(titles)
        array_list.append(arr)

    target = np.hstack(array_list)  # array([0., 0., 0., ..., 1., 1., 1., ..., 2., 2., 2.])
    print("一共加载了 %s 个标签" % target.shape)

    encoder = LabelEncoder()
    encoder.fit(target)
    encoded_target = encoder.transform(target)
    dummy_target = krs.utils.np_utils.to_categorical(encoded_target)

    return dummy_target, all_titles


def build_netword(dict, catalogue=BINARY_FLAG, embedding_size=50, max_sequence_length=30):
    if catalogue == BINARY_FLAG:
        # 配置网络结构
        model = krs.Sequential()
        model.add(krs.layers.Embedding(len(dict.items()), embedding_size, input_length=max_sequence_length))
        model.add(krs.layers.LSTM(32, dropout=0.2, recurrent_dropout=0.2))
        model.add(krs.layers.Dense(1))
        model.add(krs.layers.Activation("sigmoid"))
        model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

        return model

    elif catalogue == MULTI_FLAG:
        # 配置网络结构
        model = krs.Sequential()
        model.add(krs.layers.Embedding(len(dict.items()), embedding_size, input_length=max_sequence_length))
        model.add(krs.layers.LSTM(32, dropout=0.2, recurrent_dropout=0.2))
        model.add(krs.layers.Dense(3))
        model.add(krs.layers.Activation("softmax"))
        model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

        return model