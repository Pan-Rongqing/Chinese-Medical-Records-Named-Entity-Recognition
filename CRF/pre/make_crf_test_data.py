#!/usr/bin/env python
# -*- coding: utf-8 -*-

#
# 4 tags for character tagging: B(Begin), E(End), M(Middle), S(Single)
#将文本转化为CRF工具的输入格式
#用神经网络做实体识别时没有用到这段程序，因为特征都由BILSTM网络输出
import codecs


def character_split(input_file, output_file):
    input_data = codecs.open(input_file, 'r', 'utf-8')
    output_data = codecs.open(output_file, 'w', 'utf-8')
    for line in input_data.readlines():
        for word in line.strip():
            word = word.strip()
            if word:
                output_data.write(word + "\n")
                ##output_data.write("\n")
    input_data.close()
    output_data.close()


if __name__ == '__main__':
        #if len(sys.argv) != 3:
        #    print("pls use: python make_crf_test_data.py input output")
        #sys.exit()
        #input_file = sys.argv[1]
        #output_file = sys.argv[2]
        print("start")
        for i in range(301,401):
            input_file = "D:\\条件随机场\\CRF++-0.58\\CRF++-0.58\\example\\NER\\testdataset\\01-一般项目-format2\\一般项目-" + str(i) + ".txtoriginal.txt"
            output_file = "D:\\条件随机场\\CRF++-0.58\\CRF++-0.58\\example\\NER\\testdataset\\01-一般项目-format2\\一般项目-" + str(i) + ".txt"
            character_split(input_file, output_file)
        print("end")