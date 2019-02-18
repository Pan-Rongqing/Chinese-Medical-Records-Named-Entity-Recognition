# -*- coding: utf-8 -*-
#word2vec向量生成
from gensim.models import word2vec
import os
import gensim
import numpy as np
import codecs
import multiprocessing
from dp.data import pos_build,read_dictionary
import pickle


# 此函数作用是对初始语料进行分词处理后，作为训练模型的语料
def cut_txt(old_file):
    import jieba
    global cut_file     # 分词之后保存的文件名
    cut_file = old_file + '_cut5.txt'

    try:
        fi = open(old_file, 'r', encoding='utf-8')
    except BaseException as e:  # 因BaseException是所有错误的基类，用它可以获得所有错误类型
        print(Exception, ":", e)    # 追踪错误详细信息

    text = fi.read()  # 获取文本内容
    str_out=""
    for token in text:
        if token.isdigit():
            token="<NUM>"
        if ('\u0041' <= token <= '\u005a') or ('\u0061' <= token <= '\u007a'):
            token = "<ENG>"
        str_out=str_out+token+" "


    # new_text = jieba.cut(text, cut_all=False)  # 精确模式
    # str_out = ' '.join(new_text).replace('，', '').replace('。', '').replace('？', '').replace('！', '') \
    #     .replace('“', '').replace('”', '').replace('：', '').replace('…', '').replace('（', '').replace('）', '') \
    #     .replace('—', '').replace('《', '').replace('》', '').replace('、', '').replace('‘', '') \
    #     .replace('’', '') #去掉标点符号
    fo = open(cut_file, 'w', encoding='utf-8')
    fo.write(str_out)
    return cut_file


def model_train(train_file_name, save_model_file):  # model_file_name为训练语料的路径,save_model为保存模型名
    import logging
    # 模型训练，生成词向量
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    sentences = word2vec.Text8Corpus(train_file_name)  # 加载语料
    model = gensim.models.word2vec.Word2Vec(sentences, size=100,window=10,min_count=1,workers=multiprocessing.cpu_count())  # 训练skip-gram模型; 默认window=5
    model.save(save_model_file+".model")
    model.wv.save_word2vec_format(save_model_file+".bin",
                                  binary=True)   # 以二进制类型保存模型以便重用





if __name__=="__main__":
    cut_file=r"D:\data\corpus5.txt_cut5.txt"
    #cut_txt(r"D:\data\corpus5.txt")
    save_model_name = 'D:\data\电子病历语料库_字_100'
    #if not os.path.exists(save_model_name):  # 判断文件是否存在
    #model_train(cut_file, save_model_name)
    #else:
        #print('此训练模型已经存在，不用再次训练')




    bin="D:\data\电子病历语料库_字_100.bin"

    model = gensim.models.KeyedVectors.load_word2vec_format(bin,binary=True)
    print("done loading Word2Vec")
    vocab = model.vocab
    all=[]#
    one=[]#
    signs_dic = np.load("D:/data/dictionary/signs.npy")
    body_dic = np.load("D:/data/dictionary/body.npy")
    check_dic = np.load("D:/data/dictionary/check.npy")
    treatment_dic = np.load("D:/data/dictionary/treatment.npy")
    disease_dic = np.load("D:/data/dictionary/disease.npy")
    for mid in vocab:
        vector = list()
        for dimension in model[mid]:
            one.append(dimension)
		#实验做对比时，如果需要字典特征，则还需要在word2vec特征后面添加字典特征
        # if mid in body_dic:
        #     one.append(0.5)
        # else:
        #     one.append(-0.5)
        # if mid in signs_dic:
        #     one.append(0.5)
        # else:
        #     one.append(-0.5)
        # if mid in check_dic:
        #     one.append(0.5)
        # else:
        #     one.append(-0.5)
        # if mid in treatment_dic:
        #     one.append(0.5)
        # else:
        #     one.append(-0.5)
        # if mid in disease_dic:
        #     one.append(0.5)
        # else:
        #     one.append(-0.5)

        all.append(one)
        one=[]


    np.save("D:\data\\100dim\\np_100.npy",all)