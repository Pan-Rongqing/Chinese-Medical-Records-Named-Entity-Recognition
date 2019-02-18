import tensorflow as tf
import numpy as np
import os, argparse, time, random
from model import BiLSTM_CRF
from utils import str2bool, get_logger, get_entity
from data import read_corpus, read_dictionary, tag2label, random_embedding

## Session configuration
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # default: 0
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.2  # need ~700MB GPU memory#占用20%的显存


## hyperparameters
# #命令行参数设置
parser = argparse.ArgumentParser(description='BiLSTM-CRF for Chinese NER task')
parser.add_argument('--train_data', type=str, default='mydatapath\LFdata\BIES', help='train data source')
parser.add_argument('--test_data', type=str, default='mydatapath\LFdata\BIES', help='test data source')
parser.add_argument('--batch_size', type=int, default=64, help='#sample of each minibatch')
parser.add_argument('--epoch', type=int, default=60, help='#epoch of training')
parser.add_argument('--hidden_dim', type=int, default=300, help='#dim of hidden state')
parser.add_argument('--optimizer', type=str, default='Adam', help='Adam/Adadelta/Adagrad/RMSProp/Momentum/SGD')
parser.add_argument('--CRF', type=str2bool, default=True, help='use CRF at the top layer. if False, use Softmax')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--clip', type=float, default=5.0, help='gradient clipping')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout keep_prob')
parser.add_argument('--update_embedding', type=str2bool, default=False, help='update embedding during training')
parser.add_argument('--pretrain_embedding', type=str, default='pretrain', help='use pretrained char embedding or init it randomly')
parser.add_argument('--embedding_dim', type=int, default=100, help='random init char embedding_dim')#dim维度，只有pretrain_embedding为random时有用
parser.add_argument('--shuffle', type=str2bool, default=True, help='shuffle training data before each epoch')
parser.add_argument('--mode', type=str, default='train', help='train/test/demo')
parser.add_argument('--demo_model', type=str, default='1536659706', help='model for test and demo')
args = parser.parse_args()#PyCharm中Run - Edit Configurations - Script Parameters设置命令行参数

## get char embeddings

word2id=read_dictionary(r"D:\data\100dim\word2id_100.pkl")#data.py会生成pkl文件，是词和词向量的对应关系
if args.pretrain_embedding == 'random':
    embeddings = random_embedding(word2id, args.embedding_dim)#随机生成len（word2id）*args.embedding_dim维array
else:
    embedding_path = 'D:\\data\\100dim\\np_100.npy'#磁盘中数组的二进制格式文件
    embeddings = np.array(np.load(embedding_path), dtype='float32')#加载词向量到内存中


## read corpus and get training data
if args.mode != 'demo':
    train_path = os.path.join('.', args.train_data, 'train1.txt')#训练集文件路径拼接
    test_path = os.path.join('.', args.test_data, 'test1.txt')#测试集文件路径拼接
    train_data = read_corpus(train_path)#读训练集，为自己定义的一个函数，返回list
    test_data = read_corpus(test_path)
    test_size = len(test_data)


## paths setting
paths = {}
timestamp = str(int(time.time())) if args.mode == 'train' else args.demo_model  #返回当前时间
output_path = os.path.join('.', args.train_data+"_save", timestamp)
if not os.path.exists(output_path):
    os.makedirs(output_path)
summary_path = os.path.join(output_path, "summaries")
paths['summary_path'] = summary_path
if not os.path.exists(summary_path):
    os.makedirs(summary_path)
model_path = os.path.join(output_path, "checkpoints/")
if not os.path.exists(model_path):
    os.makedirs(model_path)
ckpt_prefix = os.path.join(model_path, "model")
paths['model_path'] = ckpt_prefix
result_path = os.path.join(output_path, "results")
paths['result_path'] = result_path
if not os.path.exists(result_path):
    os.makedirs(result_path)
log_path = os.path.join(result_path, "log.txt")
paths['log_path'] = log_path
get_logger(log_path).info(str(args))


## training model
if args.mode == 'train':
    # ckpt_file = tf.train.latest_checkpoint(model_path)
    # print(ckpt_file)
    # paths['model_path'] = ckpt_file
    model = BiLSTM_CRF(args, embeddings, tag2label, word2id, paths, config=config)
    model.build_graph()

    ## hyperparameters-tuning, split train/dev
    # dev_data = train_data[:5000]; dev_size = len(dev_data)
    # train_data = train_data[5000:]; train_size = len(train_data)
    # print("train data: {0}\ndev data: {1}".format(train_size, dev_size))
    # model.train(train=train_data, dev=dev_data)

    ## train model on the whole training data
    print("train data: {}".format(len(train_data)))#依次输出所有的train_data
    model.train(train=train_data, dev=test_data)  # use test_data as the dev_data to see overfitting phenomena

## testing model
elif args.mode == 'test':
    ckpt_file = tf.train.latest_checkpoint(model_path)
    print(ckpt_file)
    paths['model_path'] = ckpt_file
    model = BiLSTM_CRF(args, embeddings, tag2label, word2id, paths, config=config)
    model.build_graph()
    print("test data: {}".format(test_size))#输出testData
    model.test(test_data)

## demo
elif args.mode == 'demo':
    ckpt_file = tf.train.latest_checkpoint(model_path)
    print(ckpt_file)
    paths['model_path'] = ckpt_file
    model = BiLSTM_CRF(args, embeddings, tag2label, word2id, paths, config=config)
    model.build_graph()
    saver = tf.train.Saver()
    with tf.Session(config=config) as sess:
        print('============= demo =============')
        saver.restore(sess, ckpt_file)
        while(1):
            print('Please input your sentence:')
            demo_sent = input()
            if demo_sent == '' or demo_sent.isspace():
                print('See you next time!')
                break
            else:
                demo_sent = list(demo_sent.strip())
                demo_data = [(demo_sent, ['O'] * len(demo_sent))]
                tag = model.demo_one(sess, demo_data)
                print(tag)
                SIGNS, BODY, CHECK, TREAMENT,DISEASE = get_entity(tag, demo_sent)
                print('SIGNS: {}\nBODY: {}\nCHECK: {}\nTREATMENT:{}\nDISEASE:{}'.format(SIGNS, BODY, CHECK,TREAMENT,DISEASE))
