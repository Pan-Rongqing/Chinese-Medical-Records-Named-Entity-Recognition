#encoding = utf - 8
#The File is to Extract the Part-of-Speech of Text
#Jieba is used to spilt Chinese text

import nltk
import jieba.posseg as pseg
#词性解析
for i in range(241,301):
    with open("F:\Chinese Medical Records Named Entity Recognition\CRF\CCKS2017ORiginalData\CCKS2017_dataset\case_of_illness\data2\\training dataset v4\\result\病史特点\\病史特点.testt-"+str(i)+".txt",encoding="utf-8") as f:
        text = f.readlines()
        newText = ""
        for j in range(0,len(text)):
            text[j]=text[j].strip("\n")
            newText = newText + text[j]
        text = newText

    t = open("F:\Chinese Medical Records Named Entity Recognition\CRF\CCKS2017ORiginalData\CCKS2017_dataset\case_of_illness\data2\\training dataset v4\\result\病史特点\\病史特点"+str(i)+".txt","a",encoding="utf-8")
    #text = "1.患者老年女性，88岁;2.既往体健，否认药物过敏史。3.患者缘于5小时前不慎摔伤，伤及右髋部。伤后患者自感伤处疼痛，呼我院120接来我院，查左髋部X片示：左侧粗隆间骨折。给予补液等对症治疗。患者病情平稳，以左侧粗隆间骨折介绍入院。"
    res = pseg.cut(text)
    list = []

    for w in res:
        for ws in w.word:
            list.append([ws,w.flag])
            t.write(ws+"\t"+w.flag+"\n")
    print(list)

