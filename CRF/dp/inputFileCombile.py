# coding = GBK
dir=r"C:\Users\dell\PycharmProjects\CRF\dp\mydatapath\data"

area = ["一般项目", "病史特点", "诊疗经过", "出院情况"]
#四个领域的文本合并，以平衡五个种类实体的数量


if(True):
    for j in range(1,6):
        with open(dir+"\\"+area[0]+"TestTrue"+str(j)+".txt","a",encoding="utf-8") as fb:
            with open(dir+"\\"+area[1]+"TestTrue"+str(j)+".txt",encoding="utf-8") as fc:
                lines = fc.readlines()
                fb.write("\n")
                for line in lines:
                    fb.write(line)

            with open(dir+"\\"+area[2]+"TestTrue"+str(j)+".txt",encoding="utf-8") as fd:
                lines = fd.readlines()
                fb.write("\n")
                for line in lines:
                    fb.write(line)

            with open(dir+"\\"+area[3]+"TestTrue"+str(j)+".txt",encoding="utf-8") as fe:
                lines = fe.readlines()
                fb.write("\n")
                for line in lines:
                    fb.write(line)