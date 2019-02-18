#!/usr/bin/python
# -*- coding: utf-8 -*-
#CCKS中的overall评价指标
#再补齐〖eval〗_symptom，〖eval〗_disease，〖eval〗_exam，〖eval〗_treatment，〖eval〗_body
import sys

if __name__ == "__main__":
    try:
        file = open("D:\\条件随机场学习\\CRF++-0.58(可用)\\CRF++-0.58\\example\\NER\\res.csv", encoding='utf8',errors='ignore')
    except:
        print("result file is not specified, or open failed!")
        sys.exit()

    wc_of_test = 0#预测的正例个数
    wc_of_gold = 0#实际的正例个数
    wc_of_correct = 0
    flag = True
    count=0

    for line in file:
        print(count)
        count=count+1
        if line == '\n':
            continue

        s = line.strip().split(",")

        g=s[-2]
        r=s[-1]

        if r != g:
            flag = False

        if r.startswith("E",0,1)|r.startswith("S",0,1): ##in ('E', 'S'):##E、S是词结束的标志
            wc_of_test += 1
            if flag:
                wc_of_correct += 1
            flag = True

        if g.startswith("E",0,1)|g.startswith("S",0,1): ## in ('E', 'S'):
            wc_of_gold += 1

    print("WordCount from test result:", wc_of_test)
    print("WordCount from golden data:", wc_of_gold)
    print("WordCount of correct segs :", wc_of_correct)

    # 查全率
    P = wc_of_correct / float(wc_of_test)
    # 查准率，召回率
    R = wc_of_correct / float(wc_of_gold)

    print("P = %f, R = %f, F-score = %f" % (P, R, (2 * P * R) / (P + R)))