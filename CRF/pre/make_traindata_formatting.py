# coding = utf-8
# from gensim.models.word2vec import Word2Vec
import jieba.posseg as pseg

from pre import fio

# from sklearn.cluster import KMeans


##根据初始数据格式生成适合CRF工具输入的文件
##使用神经网络的实体识别没有用到这个文件
##封闭测试改为开放测试

datadir = "F:\Chinese Medical Records Named Entity Recognition\CRF\CCKS2017ORiginalData\CCKS2017_dataset\case_of_illness\data2\\training dataset v4"
area = ["一般项目" ,"病史特点","诊疗经过" , "出院情况"]
#分别计算五个实体类的最长长度，最短长度，长度总和，计数（最后两项用于求平均长度）
max=[1,1,1,1,1]
min=[1000,1000,1000,1000,1000]
sum=[0,0,0,0,0]
count=[0,0,0,0,0]


def searchIndex(word,listWordAll):
    i=0
    for w in listWordAll:
        if word==w:
            return i
        else:
            i=i+1
    return -1




class CRF_unit:


    def __init__(self):  # 初始化
        self.features = []


    # def cluster():
    #     [TrainModel, x, listWordAll] = word2vecModel()
    #     kmeans = KMeans(n_clusters=4)
    #     kmeans.fit(x)
    #     return kmeans


    def test_into_aline(self, filename):
        self.features = []
        sentences = fio.ReadFileUTF8(filename)
        for sentence in sentences:
            words = pseg.cut(sentence)#分词
            for w in words:
                for token in w.word:#切字
                    if token!="\t":
                        feature = [token, w.flag]
                        self.features.append(feature)

    def get_posTag(self, sentence):
        words = pseg.cut(sentence)
        return words

    def get_token(self, filename):
        self.features = []
        sentences = fio.ReadFileUTF8(filename)
        for sentence in sentences:
            words = self.get_posTag(sentence)  # words 包括词及其词性标签
            for w in words:
                if w.word!="\t":
                    for token in w.word:#切字
                        if token.isdigit():
                            token = "<NUM>"
                        elif ('\u0041' <= token <='\u005a') or ('\u0061' <= token <='\u007a'):
                            token = "<ENG>"
                        else:
                            token = token
                        feature=[token,w.flag,"O"]
                        self.features.append(feature)

    def read_type(self, itype):  # 将数据类型转为编码
        # itype = itype.encode("utf-8")
        if itype == "症状和体征":
            return "SIGNS"
        if itype == "检查和检验":
            return "CHECK"
        if itype == "疾病和诊断":
            return "DISEASE"
        if itype == "治疗":
            return "TREATMENT"
        if itype == "身体部位":
            return "BODY"

    def get_type(self, filename):
        sentences = fio.ReadFileUTF8(filename)
        for sentence in sentences:
            words = sentence.strip("\n").split("\t")
            start = int(words[1])
            end = int(words[2])
            #实现2-tag标注  B - I
            itype = self.read_type(words[-1])
            # self.features[start][-1] = "B-" + str(itype)
            # if (end>start):
            #     for j in range(start+1,end + 1):
            #         self.features[j][-1] = "I-"+str(itype)

            #4-tag标注   B - I - E - S
            # if start == end:
            #     self.features[start][-1] = "S-" + str(itype)
            # else:
            #     self.features[start][-1] = "B-" + str(itype)
            #     for j in range(start + 1, end):
            #         self.features[j][-1] = "I-" + str(itype)
            #     self.features[end][-1] = "E-" + str(itype)

            order=100
            if itype=="SIGNS":
                order=0
            if itype=="CHECK":
                order=1
            if itype=="DISEASE":
                order=2
            if itype =="TREATMENT":
                order=3
            if itype =="BODY":
                order=4
            leng=end-start+1

            if order in range(0,5):
                if leng<min[order]:
                    min[order]=leng
                if leng>max[order]:
                    max[order]=leng
                count[order]=count[order]+1
                sum[order]=sum[order]+leng


if __name__ == "__main__":
    # 要实现五折交叉验证

    extractor = CRF_unit()
    #x = 1;  # x=0,1,2,3


    for x in range(0,4):
        for i in range(1,301):
            filename = datadir + "/" + area[x] + "/" + area[x] + "-" + str(i) + ".txt"
            extractor.get_type(filename)  # features中的Type被修改
    print(max)
    print(min)
    print(sum)
    print(count)


    for x in range(0, 4):
        for i in range(1, 241):
            filename = datadir + "/" + area[x] + "/" + area[x] + "-" + str(i) + ".txtoriginal.txt"
            extractor.get_token(filename)  # features 已经被填充了
#
            filename = datadir + "/" + area[x] + "/" + area[x] + "-" + str(i) + ".txt"
            extractor.get_type(filename)  # features中的Type被修改
#
            filename = datadir + "/" + area[x] + "/" + area[x] + "Train1.txt"
            fio.AddTrain(extractor.features, filename)
#
        for i in range(241, 301):
            filename = datadir + "/" + area[x] + "/" + area[x] + "-" + str(i) + ".txtoriginal.txt"
            extractor.test_into_aline(filename)
#
            filename = datadir + "/" + area[x] + "/" + area[x] + "Test1.txt"
            fio.AddTest(extractor.features, filename)
#
        for i in range(241, 301):
            filename = datadir + "/" + area[x] + "/" + area[x] + "-" + str(i) + ".txtoriginal.txt"
            extractor.get_token(filename)  # features 已经被填充了
#
            filename = datadir + "/" + area[x] + "/" + area[x] + "-" + str(i) + ".txt"
            extractor.get_type(filename)  # features中的Type被修改
#
            filename = datadir + "/" + area[x] + "/" + area[x] + "TestTrue1.txt"
            fio.AddTrain(extractor.features, filename)
#
        for i in [i for i in range(1, 181)] + [i for i in range(241, 301)]:
            filename = datadir + "/" + area[x] + "/" + area[x] + "-" + str(i) + ".txtoriginal.txt"
            extractor.get_token(filename)  # features 已经被填充了
#
            filename = datadir + "/" + area[x] + "/" + area[x] + "-" + str(i) + ".txt"
            extractor.get_type(filename)  # features中的Type被修改
#
            filename = datadir + "/" + area[x] + "/" + area[x] + "Train2.txt"
            fio.AddTrain(extractor.features, filename)
#
        for i in range(181, 241):
            filename = datadir + "/" + area[x] + "/" + area[x] + "-" + str(i) + ".txtoriginal.txt"
            extractor.test_into_aline(filename)
#
            filename = datadir + "/" + area[x] + "/" + area[x] + "Test2.txt"
            fio.AddTest(extractor.features, filename)
#
        for i in range(181, 241):
            filename = datadir + "/" + area[x] + "/" + area[x] + "-" + str(i) + ".txtoriginal.txt"
            extractor.get_token(filename)  # features 已经被填充了
#
            filename = datadir + "/" + area[x] + "/" + area[x] + "-" + str(i) + ".txt"
            extractor.get_type(filename)  # features中的Type被修改
#
            filename = datadir + "/" + area[x] + "/" + area[x] + "TestTrue2.txt"
            fio.AddTrain(extractor.features, filename)
#
        for i in [i for i in range(1, 121)] + [i for i in range(181, 301)]:
            filename = datadir + "/" + area[x] + "/" + area[x] + "-" + str(i) + ".txtoriginal.txt"
            extractor.get_token(filename)  # features 已经被填充了
#
            filename = datadir + "/" + area[x] + "/" + area[x] + "-" + str(i) + ".txt"
            extractor.get_type(filename)  # features中的Type被修改
#
            filename = datadir + "/" + area[x] + "/" + area[x] + "Train3.txt"
            fio.AddTrain(extractor.features, filename)
#
        for i in range(121, 181):
            filename = datadir + "/" + area[x] + "/" + area[x] + "-" + str(i) + ".txtoriginal.txt"
            extractor.test_into_aline(filename)
#
            filename = datadir + "/" + area[x] + "/" + area[x] + "Test3.txt"
            fio.AddTest(extractor.features, filename)
#
        for i in range(121, 181):
            filename = datadir + "/" + area[x] + "/" + area[x] + "-" + str(i) + ".txtoriginal.txt"
            extractor.get_token(filename)  # features 已经被填充了
#
            filename = datadir + "/" + area[x] + "/" + area[x] + "-" + str(i) + ".txt"
            extractor.get_type(filename)  # features中的Type被修改
#
            filename = datadir + "/" + area[x] + "/" + area[x] + "TestTrue3.txt"
            fio.AddTrain(extractor.features, filename)
#
        for i in [i for i in range(1, 61)] + [i for i in range(121, 301)]:
            filename = datadir + "/" + area[x] + "/" + area[x] + "-" + str(i) + ".txtoriginal.txt"
            extractor.get_token(filename)  # features 已经被填充了
#
            filename = datadir + "/" + area[x] + "/" + area[x] + "-" + str(i) + ".txt"
            extractor.get_type(filename)  # features中的Type被修改
#
            filename = datadir + "/" + area[x] + "/" + area[x] + "Train4.txt"
            fio.AddTrain(extractor.features, filename)
#
        for i in range(61, 121):
            filename = datadir + "/" + area[x] + "/" + area[x] + "-" + str(i) + ".txtoriginal.txt"
            extractor.test_into_aline(filename)
#
            filename = datadir + "/" + area[x] + "/" + area[x] + "Test4.txt"
            fio.AddTest(extractor.features, filename)
#
        for i in range(61, 121):
            filename = datadir + "/" + area[x] + "/" + area[x] + "-" + str(i) + ".txtoriginal.txt"
            extractor.get_token(filename)  # features 已经被填充了
#
            filename = datadir + "/" + area[x] + "/" + area[x] + "-" + str(i) + ".txt"
            extractor.get_type(filename)  # features中的Type被修改
#
            filename = datadir + "/" + area[x] + "/" + area[x] + "TestTrue4.txt"
            fio.AddTrain(extractor.features, filename)
#
        for i in range(61, 301):
            filename = datadir + "/" + area[x] + "/" + area[x] + "-" + str(i) + ".txtoriginal.txt"
            extractor.get_token(filename)  # features 已经被填充了
#
            filename = datadir + "/" + area[x] + "/" + area[x] + "-" + str(i) + ".txt"
            extractor.get_type(filename)  # features中的Type被修改
#
            filename = datadir + "/" + area[x] + "/" + area[x] + "Train5.txt"
            fio.AddTrain(extractor.features, filename)
#
        for i in range(1, 61):
            filename = datadir + "/" + area[x] + "/" + area[x] + "-" + str(i) + ".txtoriginal.txt"
            extractor.test_into_aline(filename)
#
            filename = datadir + "/" + area[x] + "/" + area[x] + "Test5.txt"
            fio.AddTest(extractor.features, filename)
#
        for i in range(1, 61):
            filename = datadir + "/" + area[x] + "/" + area[x] + "-" + str(i) + ".txtoriginal.txt"
            extractor.get_token(filename)  # features 已经被填充了
#
            filename = datadir + "/" + area[x] + "/" + area[x] + "-" + str(i) + ".txt"
            extractor.get_type(filename)  # features中的Type被修改
#
            filename = datadir + "/" + area[x] + "/" + area[x] + "TestTrue5.txt"
            fio.AddTrain(extractor.features, filename)
#
#
#
#
# #
# # if __name__ == "__main__":
# #     # 要实现五折交叉验证
		#用KMeans的结果作为一维补充特征
# #     extractor = CRF_unit()
# #     #x = 1;  # x=0,1,2,3
# #     [model1,x1,listWordAll1]=word2vecModel1()
# #     [model2,x2,listWordAll2]=word2vecModel2()
# #     [model3,x3,listWordAll3]=word2vecModel3()
# #     [model4,x4,listWordAll4]=word2vecModel4()
# #     [model5,x5,listWordAll5]=word2vecModel5()
# #     km1=KMeans(n_clusters=5)
# #     km1.fit(x1)
# #     km2=KMeans(n_clusters=5)
# #     km2.fit(x2)
# #     km3=KMeans(n_clusters=5)
# #     km3.fit(x3)
# #     km4=KMeans(n_clusters=5)
# #     km4.fit(x4)
# #     km5=KMeans(n_clusters=5)
# #     km5.fit(x5)
# #
# #
# #     for x in range(0, 1):
# #         for i in range(1, 241):
# #             filename = datadir + "/" + area[x] + "/" + area[x] + "-" + str(i) + ".txtoriginal.txt"
# #             extractor.get_token(filename,km1,listWordAll1)  # features 已经被填充了
# #
# #             filename = datadir + "/" + area[x] + "/" + area[x] + "-" + str(i) + ".txt"
# #             extractor.get_type(filename)  # features中的Type被修改
# #
# #             filename = datadir + "/" + area[x] + "/" + area[x] + "Train1.txt"
# #             fio.AddTrain(extractor.features, filename)
# #
# #         for i in range(241, 301):
# #             filename = datadir + "/" + area[x] + "/" + area[x] + "-" + str(i) + ".txtoriginal.txt"
# #             extractor.test_into_aline(filename,km1,listWordAll1)
# #
# #             filename = datadir + "/" + area[x] + "/" + area[x] + "Test1.txt"
# #             fio.AddTest(extractor.features, filename)
# #
# #         for i in range(241, 301):
# #             filename = datadir + "/" + area[x] + "/" + area[x] + "-" + str(i) + ".txtoriginal.txt"
# #             extractor.get_token(filename,km1,listWordAll1)  # features 已经被填充了
# #
# #             filename = datadir + "/" + area[x] + "/" + area[x] + "-" + str(i) + ".txt"
# #             extractor.get_type(filename)  # features中的Type被修改
# #
# #             filename = datadir + "/" + area[x] + "/" + area[x] + "TestTrue1.txt"
# #             fio.AddTrain(extractor.features, filename)
# #
# #         for i in [i for i in range(1, 181)] + [i for i in range(241, 301)]:
# #             filename = datadir + "/" + area[x] + "/" + area[x] + "-" + str(i) + ".txtoriginal.txt"
# #             extractor.get_token(filename,km2,listWordAll2)  # features 已经被填充了
# #
# #             filename = datadir + "/" + area[x] + "/" + area[x] + "-" + str(i) + ".txt"
# #             extractor.get_type(filename)  # features中的Type被修改
# #
# #             filename = datadir + "/" + area[x] + "/" + area[x] + "Train2.txt"
# #             fio.AddTrain(extractor.features, filename)
# #
# #         for i in range(181, 241):
# #             filename = datadir + "/" + area[x] + "/" + area[x] + "-" + str(i) + ".txtoriginal.txt"
# #             extractor.test_into_aline(filename,km2,listWordAll2)
# #
# #             filename = datadir + "/" + area[x] + "/" + area[x] + "Test2.txt"
# #             fio.AddTest(extractor.features, filename)
# #
# #         for i in range(181, 241):
# #             filename = datadir + "/" + area[x] + "/" + area[x] + "-" + str(i) + ".txtoriginal.txt"
# #             extractor.get_token(filename,km2,listWordAll2)  # features 已经被填充了
# #
# #             filename = datadir + "/" + area[x] + "/" + area[x] + "-" + str(i) + ".txt"
# #             extractor.get_type(filename)  # features中的Type被修改
# #
# #             filename = datadir + "/" + area[x] + "/" + area[x] + "TestTrue2.txt"
# #             fio.AddTrain(extractor.features, filename)
# #
# #         for i in [i for i in range(1, 121)] + [i for i in range(181, 301)]:
# #             filename = datadir + "/" + area[x] + "/" + area[x] + "-" + str(i) + ".txtoriginal.txt"
# #             extractor.get_token(filename,km3,listWordAll3)  # features 已经被填充了
# #
# #             filename = datadir + "/" + area[x] + "/" + area[x] + "-" + str(i) + ".txt"
# #             extractor.get_type(filename)  # features中的Type被修改
# #
# #             filename = datadir + "/" + area[x] + "/" + area[x] + "Train3.txt"
# #             fio.AddTrain(extractor.features, filename)
# #
# #         for i in range(121, 181):
# #             filename = datadir + "/" + area[x] + "/" + area[x] + "-" + str(i) + ".txtoriginal.txt"
# #             extractor.test_into_aline(filename,km3,listWordAll3)
# #
# #             filename = datadir + "/" + area[x] + "/" + area[x] + "Test3.txt"
# #             fio.AddTest(extractor.features, filename)
# #
# #         for i in range(121, 181):
# #             filename = datadir + "/" + area[x] + "/" + area[x] + "-" + str(i) + ".txtoriginal.txt"
# #             extractor.get_token(filename,km3,listWordAll3)  # features 已经被填充了
# #
# #             filename = datadir + "/" + area[x] + "/" + area[x] + "-" + str(i) + ".txt"
# #             extractor.get_type(filename)  # features中的Type被修改
# #
# #             filename = datadir + "/" + area[x] + "/" + area[x] + "TestTrue3.txt"
# #             fio.AddTrain(extractor.features, filename)
# #
# #         for i in [i for i in range(1, 61)] + [i for i in range(121, 301)]:
# #             filename = datadir + "/" + area[x] + "/" + area[x] + "-" + str(i) + ".txtoriginal.txt"
# #             extractor.get_token(filename,km4,listWordAll4)  # features 已经被填充了
# #
# #             filename = datadir + "/" + area[x] + "/" + area[x] + "-" + str(i) + ".txt"
# #             extractor.get_type(filename)  # features中的Type被修改
# #
# #             filename = datadir + "/" + area[x] + "/" + area[x] + "Train4.txt"
# #             fio.AddTrain(extractor.features, filename)
# #
# #         for i in range(61, 121):
# #             filename = datadir + "/" + area[x] + "/" + area[x] + "-" + str(i) + ".txtoriginal.txt"
# #             extractor.test_into_aline(filename,km4,listWordAll4)
# #
# #             filename = datadir + "/" + area[x] + "/" + area[x] + "Test4.txt"
# #             fio.AddTest(extractor.features, filename)
# #
# #         for i in range(61, 121):
# #             filename = datadir + "/" + area[x] + "/" + area[x] + "-" + str(i) + ".txtoriginal.txt"
# #             extractor.get_token(filename,km4,listWordAll4)  # features 已经被填充了
# #
# #             filename = datadir + "/" + area[x] + "/" + area[x] + "-" + str(i) + ".txt"
# #             extractor.get_type(filename)  # features中的Type被修改
# #
# #             filename = datadir + "/" + area[x] + "/" + area[x] + "TestTrue4.txt"
# #             fio.AddTrain(extractor.features, filename)
# #
# #         for i in range(61, 301):
# #             filename = datadir + "/" + area[x] + "/" + area[x] + "-" + str(i) + ".txtoriginal.txt"
# #             extractor.get_token(filename,km5,listWordAll5)  # features 已经被填充了
# #
# #             filename = datadir + "/" + area[x] + "/" + area[x] + "-" + str(i) + ".txt"
# #             extractor.get_type(filename)  # features中的Type被修改
# #
# #             filename = datadir + "/" + area[x] + "/" + area[x] + "Train5.txt"
# #             fio.AddTrain(extractor.features, filename)
# #
# #         for i in range(1, 61):
# #             filename = datadir + "/" + area[x] + "/" + area[x] + "-" + str(i) + ".txtoriginal.txt"
# #             extractor.test_into_aline(filename,km5,listWordAll5)
# #
# #             filename = datadir + "/" + area[x] + "/" + area[x] + "Test5.txt"
# #             fio.AddTest(extractor.features, filename)
# #
# #         for i in range(1, 61):
# #             filename = datadir + "/" + area[x] + "/" + area[x] + "-" + str(i) + ".txtoriginal.txt"
# #             extractor.get_token(filename,km5,listWordAll5)  # features 已经被填充了
# #
# #             filename = datadir + "/" + area[x] + "/" + area[x] + "-" + str(i) + ".txt"
# #             extractor.get_type(filename)  # features中的Type被修改
# #
# #             filename = datadir + "/" + area[x] + "/" + area[x] + "TestTrue5.txt"
# #             fio.AddTrain(extractor.features, filename)