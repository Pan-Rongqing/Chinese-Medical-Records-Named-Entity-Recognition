import tensorflow as tf
import numpy as np
#构建词典文件，除身体部位外的四类实体词典构建是利用来自网络的词库，身体部位这一实体因为没有在网络中找到合适的词库，所以使用训练集中的实体构建
with open(r"C:\Users\dell\Desktop\\pku_data\signs_dic.csv",encoding="utf-8") as f:
    text=f.readlines()
    signs_dic=[]
    for line in text:
        for token in line.split(",")[0]:
            if token.isdigit():
                token = "<NUM>"
            if ('\u0041' <= token <='\u005a') or ('\u0061' <= token <='\u007a'):
                token = "<ENG>"
            if token not in signs_dic:
                signs_dic.append(token)
print(signs_dic)

treatment_dic=[]
with open(r"C:\Users\dell\Desktop\\pku_data\treatment_dic.csv",encoding="utf-8") as f:
    text=f.readlines()
    for line in text:
        for token in line.split(",")[0]:
            if token.isdigit():
                token = "<NUM>"
            if ('\u0041' <= token <='\u005a') or ('\u0061' <= token <='\u007a'):
                token = "<ENG>"
            if token not in treatment_dic:
                treatment_dic.append(token)
print(treatment_dic)

with open(r"C:\Users\dell\Desktop\\pku_data\medicine_dic.csv",encoding="utf-8") as f:
    text=f.readlines()
    for line in text:
        for token in line.split(",")[0]:
            if token.isdigit():
                token = "<NUM>"
            if ('\u0041' <= token <='\u005a') or ('\u0061' <= token <='\u007a'):
                token = "<ENG>"
            if token not in treatment_dic:
                treatment_dic.append(token)
print(medicine_dic)

with open(r"C:\Users\dell\Desktop\\pku_data\check_dic.csv",encoding="utf-8") as f:
    text=f.readlines()
    check_dic=[]
    for line in text:
        for token in line.split(",")[0]:
            if token.isdigit():
                token = "<NUM>"
            if ('\u0041' <= token <='\u005a') or ('\u0061' <= token <='\u007a'):
                token = "<ENG>"
            if token not in check_dic:
                check_dic.append(token)
print(check_dic)

with open(r"C:\Users\dell\Desktop\\pku_data\disease_dic.csv",encoding="utf-8") as f:
    text=f.readlines()
    disease_dic=[]
    for line in text:
        for token in line.split(",")[0]:
            if token.isdigit():
                token = "<NUM>"
            if ('\u0041' <= token <='\u005a') or ('\u0061' <= token <='\u007a'):
                token = "<ENG>"
            if token not in disease_dic:
                disease_dic.append(token)
print(disease_dic)


with open("C:/Users/dell/PycharmProjects/CRF/dp/mydatapath/LFdata/train5.txt",encoding="utf-8") as f:
    text = f.readlines()
    body_dic = []
    for line in text:
        if line!="\n":
            (token,pos,tag) = line.strip().split("\t")
            if tag == "B-BODY" or tag =="I-BODY":
                if token not in body_dic:
                    body_dic.append(token)
print(body_dic)

np.save("D:/data/dictionary/body.npy",body_dic)
np.save("D:/data/dictionary/disease.npy",disease_dic)
np.save("D:/data/dictionary/check.npy",check_dic)
np.save("D:/data/dictionary/treatment.npy",treatment_dic)
np.save("D:/data/dictionary/signs.npy",signs_dic)