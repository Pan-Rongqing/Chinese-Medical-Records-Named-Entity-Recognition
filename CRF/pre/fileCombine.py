#无法解决神经网络输出后中文乱码问题，故将test文件信息和预测标签做人工拼接

for i in range(1,6):
    with open("D:\CRF\CRF++-0.58\CRF++-0.58\example\CCKS(3)\诊疗经过\P" + str(i) + ".txt","r",encoding="utf-8") as f:
        label = f.readlines()
        listLabel = []
        for line in label:
            listLabel.append(line)

    with open("D:\CRF\CRF++-0.58\CRF++-0.58\example\CCKS(3)\诊疗经过\诊疗经过Test" + str(i) + ".txt","r",encoding="utf-8") as f2:
        tokens = f2.readlines()
        listToken = []
        for line in tokens:
            listToken.append(line.split("\t")[0])


    f3=open("D:\CRF\CRF++-0.58\CRF++-0.58\example\CCKS(3)\诊疗经过\诊疗经过Predict" + str(i) + ".txt","a",encoding="utf-8")
    for i in range(0,len(listLabel)):
        if listToken[i] != "\n":
            T=listToken[i]
            L=listLabel[i]
            f3.write(listToken[i] + "\t" + listLabel[i])
        else:
            f3.write("\n")

    f3.flush()
    f3.close()
