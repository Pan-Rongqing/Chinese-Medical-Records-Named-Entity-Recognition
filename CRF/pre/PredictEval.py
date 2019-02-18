#四个实体：BODY-SIGNS-CHECK-DISEASE
#评价函数
import sys
#检索两个列表的子集
def search_same(A,B):
    start = 0
    AB = []
    for x in A:
        for i in range(start,len(B)):
            if( x == B[i]):
                AB.append(x)
                start = i + 1 ##
                break
            elif x>B[i]:
                start = i + 1
            else:
                pass
    return AB

f1 = open(r"F:\Chinese Medical Records Named Entity Recognition\CRF\dp\mydatapath\data_save\1536057964\results\label_60","r",encoding="utf-8")
f2 = f1
listTrueBODY=[]
listTrueSIGNS=[]
listTrueCHECK=[]
listTrueDISEASE=[]
listTrueTREATMENT=[]
listPredictBODY=[]
listPredictSIGNS=[]
listPredictCHECK=[]
listPredictDISEASE=[]
listPredictTREATMENT=[]
splitCode = " "   #分隔符



text1 = f1.readlines()
word = ""
for j in range(0,len(text1)):
    if text1[j]!="\n":
        if text1[j].split(splitCode)[-2].startswith("B",0,1):
            word = word+text1[j].split(splitCode)[0]
        elif text1[j].split(splitCode)[-2].startswith("I",0,1):
            word = word+text1[j].split(splitCode)[0]
        elif text1[j].split(splitCode)[-2].startswith("E",0,1):
            word = word+text1[j].split(splitCode)[0]
            area = text1[j].split(splitCode)[-2].strip().split("-")
            if area[1]=="BODY":
                listTrueBODY.append(word)
            elif area[1]=="SIGNS":
                listTrueSIGNS.append(word)
            elif area[1]=="CHECK":
                listTrueCHECK.append(word)
            elif area[1]=="TREATMENT":
                listTrueTREATMENT.append(word)
            else :
                listTrueDISEASE.append(word)
            word = ""
        elif text1[j].split(splitCode)[-2].startswith("S",0,1):
            word = word + text1[j].split(splitCode)[0]
            area = text1[j].split(splitCode)[-2].strip().split("-")
            if area[1] == "BODY":
                listTrueBODY.append(word)
            elif area[1] == "SIGNS":
                listTrueSIGNS.append(word)
            elif area[1] == "CHECK":
                listTrueCHECK.append(word)
            elif area[1] == "TREATMENT":
                listTrueTREATMENT.append(word)
            else:
                listTrueDISEASE.append(word)
            word = ""
        else:
            continue


text2 = text1
word = ""
for j in range(0, len(text2)):
    if text2[j]!="\n":
        if text2[j].split(splitCode)[-1].startswith("B", 0, 1):
            word = word + text2[j].split(splitCode)[0]
        elif text2[j].split(splitCode)[-1].startswith("I", 0, 1):
            word = word + text2[j].split(splitCode)[0]


        elif text2[j].split(splitCode)[-1].startswith("E", 0, 1):
            word = word + text2[j].split(splitCode)[0]
            area = text2[j].split(splitCode)[-1].strip().split("-")
            if area[-1] == "BODY":
                listPredictBODY.append(word)
            elif area[-1] == "SIGNS":
                listPredictSIGNS.append(word)
            elif area[-1] == "CHECK":
                listPredictCHECK.append(word)
            elif area[-1] == "TREATMENT":
                listPredictTREATMENT.append(word)
            else:
                listPredictDISEASE.append(word)
            word = ""
        elif text2[j].split(splitCode)[-1].startswith("S", 0, 1):
            word = word + text2[j].split(splitCode)[0]
            area = text2[j].split(splitCode)[-1].strip().split("-")
            if area[-1] == "BODY":
                listPredictBODY.append(word)
            elif area[-1] == "SIGNS":
                listPredictSIGNS.append(word)
            elif area[-1] == "CHECK":
                listPredictCHECK.append(word)
            elif area[-1] == "TREATMENT":
                listPredictTREATMENT.append(word)
            else:
                listPredictDISEASE.append(word)
            word = ""
        else:
            continue


TrueBODY=sorted(listTrueBODY)
TrueSIGNS=sorted(listTrueSIGNS)
TrueCHECK=sorted(listTrueCHECK)
TrueDISEASE=sorted(listTrueDISEASE)
TrueTREATMENT=sorted(listTrueTREATMENT)
PredictBODY=sorted(listPredictBODY)
PredictSIGNS=sorted(listPredictSIGNS)
PredictCHECK=sorted(listPredictCHECK)
PredictDISEASE=sorted(listPredictDISEASE)
PredictTREATMENT=sorted(listPredictTREATMENT)

#print("TrueDISEASE")
#print(TrueDISEASE)
#print("PredictDISEASE")
#print(PredictDISEASE)
#
#print("TrueSIGNS")
#print(TrueSIGNS)
#print("PredictSIGNS")
#print(PredictSIGNS)
#
#print(search_same(TrueDISEASE,PredictDISEASE))
# print(search_same(TrueSIGNS,PredictSIGNS))
#
#
# print("TrueBODY")
# print(TrueBODY)
# print("PredictBODY")
# print(PredictBODY)
#
# print(search_same(TrueBODY,PredictBODY))



print("BODY:")
wc_of_test = len(PredictBODY)
wc_of_gold = len(TrueBODY)
wc_of_correct = len(search_same(TrueBODY,PredictBODY))
if wc_of_gold!=0 and wc_of_test!=0:
    P = wc_of_correct / float(wc_of_test)
    R = wc_of_correct / float(wc_of_gold)
    print("P = %f, R = %f, F-score = %f" % (P, R, (2 * P * R) / (P + R)))


print("DISEASE:")
wc_of_test = len(PredictDISEASE)
wc_of_gold = len(TrueDISEASE)
wc_of_correct = len(search_same(TrueDISEASE,PredictDISEASE))
if wc_of_gold!=0 and wc_of_test!=0:
    P = wc_of_correct / float(wc_of_test)
    R = wc_of_correct / float(wc_of_gold)
    print("P= %f" % P)
    print("R= %f" % R)
    if P!=0 and R !=0:
        print("P = %f, R = %f, F-score = %f" % (P, R, (2 * P * R) / (P + R)))

print("CHECK:")
wc_of_test = len(PredictCHECK)
wc_of_gold = len(TrueCHECK)
wc_of_correct = len(search_same(TrueCHECK,PredictCHECK))
if wc_of_gold!=0 and wc_of_test!=0:
    P = wc_of_correct / float(wc_of_test)
    R = wc_of_correct / float(wc_of_gold)



    print("P = %f, R = %f, F-score = %f" % (P, R, (2 * P * R) / (P + R)))

print("SIGNS:")
wc_of_test = len(PredictSIGNS)
wc_of_gold = len(TrueSIGNS)
wc_of_correct = len(search_same(TrueSIGNS,PredictSIGNS))
if wc_of_gold!=0 and wc_of_test!=0:
    P = wc_of_correct / float(wc_of_test)
    R = wc_of_correct / float(wc_of_gold)
    print("P = %f, R = %f, F-score = %f" % (P, R, (2 * P * R) / (P + R)))



print("TREATMENT:")
wc_of_test = len(PredictTREATMENT)
wc_of_gold = len(TrueTREATMENT)
wc_of_correct = len(search_same(TrueTREATMENT,PredictTREATMENT))
if wc_of_gold!=0 and wc_of_test!=0:
    P = wc_of_correct / float(wc_of_test)
    R = wc_of_correct / float(wc_of_gold)
    print("P = %f, R = %f, F-score = %f" % (P, R, (2 * P * R) / (P + R)))


print("overall:")
wc_of_test = len(PredictCHECK)+len(PredictDISEASE)+len(PredictBODY)+len(PredictSIGNS)+len(PredictTREATMENT)
wc_of_gold = len(TrueCHECK)+len(TrueDISEASE)+len(TrueBODY)+len(TrueSIGNS)+len(TrueTREATMENT)
wc_of_correct = len(search_same(TrueCHECK,PredictCHECK))+len(search_same(TrueDISEASE,PredictDISEASE))+len(search_same(TrueBODY,PredictBODY))+len(search_same(TrueSIGNS,PredictSIGNS))+len(search_same(TrueTREATMENT,PredictTREATMENT))
if wc_of_gold!=0 and wc_of_test!=0:
    P = wc_of_correct / float(wc_of_test)
    R = wc_of_correct / float(wc_of_gold)
    print("P = %f, R = %f, F-score = %f" % (P, R, (2 * P * R) / (P + R)))

