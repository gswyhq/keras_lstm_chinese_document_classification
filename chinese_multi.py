# -*- coding: utf-8 -*-

import sys
import tensorflow as tf
import jieba as jb
import numpy as np
from keras.models import load_model
from tensorflow.contrib.learn import  preprocessing

import utils

max_sequence_length = 30
embedding_size = 50

SAVE_MODEL_FILE = 'health_and_tech_design.h5'
SAVE_VOCAB_FILE = 'vocab.pickle'

def train():
    target, titles= utils.load_data_label()

    # 标题分词
    titles = [".".join(jb.cut(t, cut_all=True)) for t in titles]

    # word2vec 词袋化
    # min_frequency： 设定一个频率，比如出现次数小于1的不要
    vocab_processor = preprocessing.VocabularyProcessor(max_sequence_length, min_frequency=1)
    text_processed = np.array(list(vocab_processor.fit_transform(titles)))

    # 保存词汇表
    vocab_processor.save(SAVE_VOCAB_FILE)

    # 读取标签
    dict = vocab_processor.vocabulary_._mapping
    sorted_vocab = sorted(dict.items(), key = lambda x : x[1])

    # 配置网络结构
    model = utils.build_netword(catalogue=utils.MULTI_FLAG, dict=dict, embedding_size=embedding_size, max_sequence_length=max_sequence_length)


    # 训练模型
    model.fit(text_processed, target, batch_size=512, epochs=10, )

    # 同时保存model和权重的方式：
    model.save(SAVE_MODEL_FILE)

def test_classification(sen):

    # 加载预训练的模型
    # tips：载入整个模型结构时，若模型训练时有自定义loss或metrics，则载入时会报类似错：Unknown metric function:my_loss （此处my_loss是一个自定义函数），则加载模型时需要指定custom_objects参数：
    #model = load_model('model.h5'，{'my_loss': my_loss})

    model = load_model(SAVE_MODEL_FILE)

    # 预测样本

    sen_prosessed = " ".join(jb.cut(sen, cut_all=True))
    
    # 加载词汇表
    vocab_processor = preprocessing.VocabularyProcessor.restore(SAVE_VOCAB_FILE)

    sen_prosessed = vocab_processor.transform([sen_prosessed])
    sen_prosessed = np.array(list(sen_prosessed))
    # print(sen_prosessed)
    result = model.predict(sen_prosessed)

    catalogue = list(result[0]).index(max(result[0]))
    if max(result[0]) > 0.8:
        if catalogue == 0:
            print("这是一篇关于健康的文章")
        elif catalogue == 1:
            print("这是一篇关于科技的文章")
        elif catalogue == 2:
            print("这是一篇关于设计的文章")
        else:
            print("这篇文章没有可信分类")

def main():
    if len(sys.argv) > 1 and sys.argv[1] == 'train':
        train()
        return
    elif len(sys.argv) > 1:
        sen = sys.argv[1]
    else:
        sen = "做好商业设计需要学习的小技巧"

    test_classification(sen)

if __name__ == '__main__':
    main()

