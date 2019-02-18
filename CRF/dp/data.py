import sys, pickle, os, random
import numpy as np

#tags,BIESN
tag2label = {"O":0,
             "B-SIGNS":1,"I-SIGNS":2,"E-SIGNS":3,"S-SIGNS":4,
             "B-BODY":5,"I-BODY":6,"E-BODY":7,"S-BODY":8,
             "B-CHECK":9,"I-CHECK":10,"E-CHECK":11,"S-CHECK":12,
             "B-TREATMENT":13,"I-TREATMENT":14,"E-TREATMENT":15,"S-TREATMENT":16,
             "B-DISEASE":17,"I-DISEASE":18,"E-DISEASE":19,"S-DISEASE":20
}

#tags:BIO
# tag2label = {"O":0,
#              "B-SIGNS":1,"I-SIGNS":2,
#              "B-BODY":3,"I-BODY":4,
#              "B-CHECK":5,"I-CHECK":6,
#              "B-TREATMENT":7,"I-TREATMENT":8,
#              "B-DISEASE":9,"I-DISEASE":10
# }

def pos_build(corpus_path):
    data = read_corpus(corpus_path)
    pos2id={}
    for sent_,pos_, tag_ in data:
         for pos in pos_:
             if pos not in pos2id:
                 pos2id[pos] = len(pos2id)+1#赋编号
    pos2id["unknown"]=len(pos2id)+1

    with open("D:\data\pos2id.pkl", 'wb') as fw:
        pickle.dump(pos2id, fw)

    print("len(pos2id)="+str(len(pos2id)))
     #将word2id保存到文件中，实现持久化存储


def read_corpus(corpus_path):
    """
    read corpus and return the list of samples
    :param corpus_path:
    :return: data
    """
    data = []
    with open(corpus_path, encoding='utf-8') as fr:
        lines = fr.readlines()
    sent_,tag_ = [], []
    i=1
    for line in lines:
        if line != '\n':
            [char,pos,label] = line.strip("\n").split("\t")
            print(i)
            i=i+1
            sent_.append(char)
            tag_.append(label)
            #data.append((char, label))
        else:
            data.append((sent_, tag_))
            print(i)
            i=i+1
            sent_, tag_ = [],[]
    #data是一个列表，里面多个tuple，每个tuple是字列表和序列列表的组合
    return data


def vocab_build(vocab_path, corpus_path, min_count):
    # """
    #
    # :param vocab_path:
    # :param corpus_path:
    # :param min_count:
    # :return:
    # """
    # data = read_corpus(corpus_path)
    # word2id = {}
    # for sent_, tag_ in data:
    #     for word in sent_:
    #         if word.isdigit():
    #             word = '<NUM>'
    #         if ('\u0041' <= word <='\u005a') or ('\u0061' <= word <='\u007a'):
    #             word = '<ENG>'
    #         if word not in word2id:
    #             word2id[word] = [len(word2id)+1, 1]#词id，初始词频置为1
    #         else:
    #             word2id[word][1] += 1#词频计数+1
    # low_freq_words = []#低频词列表
    # for word, [word_id, word_freq] in word2id.items():
    #     if word_freq < min_count and word != '<NUM>' and word != '<ENG>':#词频小于最低阈值，且词不为数字和英文字符
    #         low_freq_words.append(word)#加入低频词列表
    # for word in low_freq_words:
    #     del word2id[word]
    #
    # #对留下来的词重新赋id
    # new_id = 1
    # for word in word2id.keys():
    #     word2id[word] = new_id
    #     new_id += 1
    # word2id['<UNK>'] = new_id
    # word2id['<PAD>'] = 0
    #
    # print("len(word2id)="+str(len(word2id)))
    # #将word2id保存到文件中，实现持久化存储

    bin = "D:\data\电子病历语料库_字_100.bin"#bin文件是word2vec向量文件
    import gensim
    model = gensim.models.KeyedVectors.load_word2vec_format(bin,binary=True)
    vocab=model.vocab
    word2id={}
    id=1
    for mid in vocab:
        word2id[mid]=id
        id+=1
    word2id["<UNK>"]=0
    with open(vocab_path, 'wb') as fw:
        pickle.dump(word2id, fw)#pkl文件实现词和词向量文件中的对应关系


def sentence2id(sent, word2id):
    """

    :param sent:
    :param word2id:
    :return:
    """
    sentence_id = []
    for word in sent:
        if word.isdigit():#检测字符是否全是数字
            word = '<NUM>'
        elif ('\u0041' <= word <= '\u005a') or ('\u0061' <= word <= '\u007a'):##十六进制数，From "a" to "z" and From "A" to "Z"
            word = '<ENG>'
        if word not in word2id:
             word = '<UNK>'
        sentence_id.append(word2id[word])
    return sentence_id


def read_dictionary(vocab_path):
    """
    :param vocab_path:
    :return:
    """
    vocab_path = os.path.join(vocab_path)
    with open(vocab_path, 'rb') as fr:
        word2id = pickle.load(fr)
    print('vocab_size:', len(word2id))
    return word2id


def random_embedding(vocab, embedding_dim):
    """

    :param vocab:
    :param embedding_dim:
    :return:
    """
    embedding_mat = np.random.uniform(-0.25, 0.25, (len(vocab), embedding_dim))#在【-0.25，0.25）区间内采样，输出len(vocab)*embedding_dim的tuple
    embedding_mat = np.float32(embedding_mat)#数据转为float32
    return embedding_mat


def pad_sequences(sequences, pad_mark=0):
    """
    序列填充
    :param sequences:
    :param pad_mark:填充字符
    :return:
    """
    max_len = max(map(lambda x : len(x),sequences))
    seq_list, seq_len_list = [], []
    for seq in sequences:
        #seq=list(seq)  #int is not iterable
        seq = list(seq)#切字
        seq_ = seq[:max_len] + [pad_mark] * max(max_len - len(seq), 0)#不足max_len部分用0填充
        seq_list.append(seq_)#
        seq_len_list.append(min(len(seq), max_len))
    return seq_list, seq_len_list#seq_list即word_ids


def batch_yield(data, batch_size, vocab, tag2label, shuffle=False):#建立词和ID的映射

    if shuffle:
        random.shuffle(data)#随机打乱一个数组

    seqs, labels = [], []
    for (sent_, tag_) in data:
        #sent_是一个列表（文本序列），tag也是一个列表（文本序列对应的标签）
        #使用sentence2id将文本序列映射为数值序列，为自己定义的一个文本处理函数
        sent_ = sentence2id(sent_, vocab)#vocab是word2id文件，形式是【1，4，2，3，4】
        #使用tag2label将tag映射为数值，为自己定义的一个文本处理函数
        label_ = [tag2label[tag] for tag in tag_]#形式是【0，1，2，3】，是标签对应的序号
        if len(seqs) == batch_size:
            yield seqs, labels
            seqs, labels = [], []

        seqs.append(sent_)
        labels.append(label_)

    if len(seqs) != 0:
        yield seqs, labels

# if __name__=="__main__":
#     word2id=vocab_build(vocab_path=r"C:\Users\dell\Desktop\zh-NER-TF-master\zh-NER-TF-master\mydatapath\word2id.pkl",
#                         corpus_path=r"C:\Users\dell\Desktop\zh-NER-TF-master\zh-NER-TF-master\mydatapath\病史特点Train1.txt",
#                         min_count=3)



if __name__ == "__main__":
    # train_path = os.path.join('.', "mydatapath\data3", 'train_data')D:\data\
    # print(train_path)
    # train_data=read_corpus(train_path)
    # print(len(train_data))
    # print(train_data[len(train_data)-1])
    word2id = vocab_build(vocab_path=r"D:\data\100dim\word2id_1002.pkl",
                         corpus_path=r"D:\data\电子病历语料库_字_100.bin",
                         min_count=3)