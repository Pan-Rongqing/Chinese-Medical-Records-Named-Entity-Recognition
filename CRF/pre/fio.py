# coding:utf-8
import sys
import os
import codecs

datadir = "D:\条件随机场学习\一个中文的电子病例测评相关的数据CCKS2017\CCKS2017\CCKS2017_dataset\case_of_illness\data2\\training dataset v4"


def ReadFile(file):
    f = open(file, 'r')
    lines = f.readlines()
    f.close()

    return [line.rstrip('\r\n') for line in lines]


def ReadFileUTF8(file):
    f = codecs.open(file, encoding="utf-8")
    lines = f.readlines()
    f.close()

    return [line.rstrip() for line in lines]


def SaveFeatures(features, file, linetag="\n"):
    #sys.reload()
    #sys.setdefaultencoding('utf8')

    f = open(file, "w")
    for [token, tag] in features:
        f.write(token + " " + tag)
        f.write(linetag)
    f.flush()
    f.close()


def AddTrain(features, file, linetag="\n"):
    #sys.reload()
    #sys.setdefaultencoding('utf8')

    f = open(file, "a",encoding="utf-8")
    for [token, pos,tag] in features:
        if token!=" ":
            f.write(token + "\t" + pos + "\t"  + tag)
            f.write(linetag)
        if token in ["。","!","?"]:
            f.write(linetag)

    f.flush()
    f.close()


def AddTest(features, file, linetag="\n"):
    #reload(sys)
    #sys.setdefaultencoding('utf8')

    f = open(file, "a",encoding="utf-8")
    for [token, pos] in features:
        if token!=" ":
            f.write(token + "\t" + pos + "\t" )
            f.write(linetag)
        if token in ["。","!","?"]:
            f.write(linetag)
    f.flush()
    f.close()


if __name__ == '__main__':
    lines = ReadFileUTF8(datadir + '/病史特点/病史特点-1.txtoriginal.txt');
    for line in lines:
        print(line)
