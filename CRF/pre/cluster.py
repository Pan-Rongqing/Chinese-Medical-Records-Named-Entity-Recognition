from sklearn.cluster import KMeans

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from gensim.models.word2vec import Word2Vec
import jieba
import logging
import gensim
from gensim.models.word2vec import LineSentence
import re
import jieba.posseg as pseg

area = ["病史特点", "出院情况", "一般项目", "诊疗经过"]

def searchIndex(word,listWordAll):
    i=0
    for w in listWordAll:
        if word==w:
            return i
        else:
            i=i+1
    return -1




list=[]
listWordAll=[]
for i in range(1,241):
    for j in range(0,4)
    with open(r"D:\CRF\一个中文的电子病例测评相关的数据CCKS2017\CCKS2017\CCKS2017_dataset\case_of_illness\data2\training dataset v4\诊疗经过\诊疗经过-"+str(i)+".txtoriginal.txt",encoding="utf-8") as f:
        text = f.read()
        listword=[]
        seg=pseg.cut(text)
        for [word,flag] in seg:
            listword.append(word)
            listWordAll.append(word)
        list.append(listword)
print(list)


TrainModel =Word2Vec(min_count=1,size=200,seed=2,hs=1,sg=1,workers=1)#min_count是词频次阈值，size是词向量维度
TrainModel.build_vocab(list)
TrainModel.train(list,total_examples=TrainModel.corpus_count,epochs = TrainModel.iter)
x=np.ones((len(listWordAll),200))


i=0
for word in listWordAll:
    x[i]=TrainModel[word]
    i=i+1


##肘部法测试聚类合适的簇数，转折点为k=4
aa=[]
K = range(1, 10)
for k in range(1,10):
    kmeans=KMeans(n_clusters=k)
    kmeans.fit(x)
    aa.append(sum(np.min(cdist(x, kmeans.cluster_centers_, 'euclidean'),axis=1))/x.shape[0])
plt.figure()
plt.plot(np.array(K), aa, 'bx-')
plt.show()

# listWord=[]
#
# with open(r"D:\CRF\一个中文的电子病例测评相关的数据CCKS2017\CCKS2017\CCKS2017_dataset\case_of_illness\data2\training dataset v4\病史特点\病史特点-1.txtoriginal.txt",encoding="utf-8") as f:
#     con = f.read()
#     seg = pseg.cut(con)
#     for [w,flag] in seg:
#         listWord.append(w)
#
#
#
# print(searchIndex("女性",listWordAll))
#
# kmeans = KMeans(n_clusters=4)
# kmeans.fit(x)
# print("Construct model successfully")
#
# # for label in kmeans.labels_:
# #     print(label)
# # print(kmeans.labels_[searchIndex("1.",listWordAll)])
#
# for w in listWord:
#     index = searchIndex(w, listWordAll)
#     print(w)
#     print(str(kmeans.labels_[index]))
# for label in kmeans.labels_:
#      print(label)

     #print(kmeans.labels_)
