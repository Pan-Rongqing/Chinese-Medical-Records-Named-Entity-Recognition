import numpy as np
import os, time, sys
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell
from tensorflow.contrib.crf import crf_log_likelihood
from tensorflow.contrib.crf import viterbi_decode
from data import pad_sequences, batch_yield,tag2label
from utils import get_logger
from eval import conlleval


class BiLSTM_CRF(object):
    def __init__(self, args, embeddings, tag2label, vocab, paths, config):
        self.batch_size = args.batch_size
        self.epoch_num = args.epoch
        self.hidden_dim = args.hidden_dim
        self.embeddings = embeddings
        self.CRF = args.CRF#True,顶端添加CRF层
        self.update_embedding = args.update_embedding
        self.dropout_keep_prob = args.dropout
        self.optimizer = args.optimizer#优化器，采用SGD
        self.lr = args.lr#学习率
        self.clip_grad = args.clip
        self.tag2label = tag2label#标签序号对应
        self.num_tags = len(tag2label)#标签数量
        self.vocab = vocab#训练集词汇表
        self.shuffle = args.shuffle
        self.model_path = paths['model_path']
        self.summary_path = paths['summary_path']
        self.logger = get_logger(paths['log_path'])
        self.result_path = paths['result_path']
        self.config = config
        self.pos_embeddings=[1,2,3,4,5]

    def build_graph(self):
        self.add_placeholders()
        self.lookup_layer_op()
        self.biLSTM_layer_op()#bilstm层
        self.softmax_pred_op()
        self.loss_op()##crf层
        self.trainstep_op()
        self.init_op()

    def add_placeholders(self):#占位
        #创建需要的占位符，分别为输入占位符，标签占位符，序列长度占位符，dropout占位符，和学习率占位符
        self.word_ids = tf.placeholder(tf.int32, shape=[None, None], name="word_ids")
        self.labels = tf.placeholder(tf.int32, shape=[None, None], name="labels")
        self.sequence_lengths = tf.placeholder(tf.int32, shape=[None], name="sequence_lengths")
        self.dropout_pl = tf.placeholder(dtype=tf.float32, shape=[], name="dropout")
        self.lr_pl = tf.placeholder(dtype=tf.float32, shape=[], name="lr")

    def lookup_layer_op(self):
        #创建嵌入层
        with tf.variable_scope("words"):
            #Variables created here will be named "words/_word_embeddings","words/word_embeddings"
            _word_embeddings = tf.Variable(self.embeddings,
                                           dtype=tf.float32,
                                           trainable=self.update_embedding,
                                           name="_word_embeddings")#词向量
            word_embeddings = tf.nn.embedding_lookup(params=_word_embeddings,
                                                     ids=self.word_ids,
                                                     name="word_embeddings")#词空间寻找对应的向量
        self.word_embeddings =  tf.nn.dropout(word_embeddings, self.dropout_pl)#按dropout_pl的概率舍弃神经元

    def biLSTM_layer_op(self):
        with tf.variable_scope("bi-lstm"):
            #创建前向LSTM网络和后向LSTM网络
            cell_fw = LSTMCell(self.hidden_dim)
            cell_bw = LSTMCell(self.hidden_dim)
            (output_fw_seq, output_bw_seq), _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell_fw,#前向RNN
                cell_bw=cell_bw,#后向RNN
                inputs=self.word_embeddings,#输入
                sequence_length=self.sequence_lengths,#输入序列的实际长度
                dtype=tf.float32)#初始化和输出的数据类型
            #将两个方向的网络输出连接起来
            output = tf.concat([output_fw_seq, output_bw_seq], axis=-1)#使用concat（）函数将两个按行拼接
            output = tf.nn.dropout(output, self.dropout_pl)
        #将结果输入隐含层，
        with tf.variable_scope("proj"):
            #get_variable的reuse参数默认为False，则创建新的变量，否则重用已经创建的变量
            W = tf.get_variable(name="W",#参数W
                                shape=[2 * self.hidden_dim, self.num_tags],
                                initializer=tf.contrib.layers.xavier_initializer(),
                                dtype=tf.float32)

            b = tf.get_variable(name="b",#偏置项b
                                shape=[self.num_tags],
                                initializer=tf.zeros_initializer(),
                                dtype=tf.float32)

            s = tf.shape(output)#得到output的尺寸
            output = tf.reshape(output, [-1, 2*self.hidden_dim])#output的一维值不定，根据二维值确定
            pred = tf.matmul(output, W) + b#计算最终输出的预测值，用于之后的交叉熵计算

            self.logits = tf.reshape(pred, [-1, s[1], self.num_tags]) 

    def loss_op(self):#顶层;CRF层，并定义损失函数
        if self.CRF:
            #添加crf层
            log_likelihood, self.transition_params = crf_log_likelihood(inputs=self.logits,#输入
                                                                   tag_indices=self.labels,#真实标签
                                                                   sequence_lengths=self.sequence_lengths)#每个序列的长度
            #定义损失函数
            self.loss = -tf.reduce_mean(log_likelihood)

        else:
            #softmax转换，定义交叉熵
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits,
                                                                    labels=self.labels)
            #构建序列长度的mask标志
            mask = tf.sequence_mask(self.sequence_lengths)
            #将使losses(m维)矩阵仅保留与mask中“True”元素同下标的部分。
            losses = tf.boolean_mask(losses, mask)
            #求交叉熵，交叉熵是实际输出与期望输出的距离
            self.loss = tf.reduce_mean(losses)

        tf.summary.scalar("loss", self.loss)#用来显示标量信息

    def softmax_pred_op(self):
        if not self.CRF:
            self.labels_softmax_ = tf.argmax(self.logits, axis=-1)
            self.labels_softmax_ = tf.cast(self.labels_softmax_, tf.int32)#类型做转换

    def trainstep_op(self):#各种反向传播训练方法，默认是梯度下降算法
        #使用优化器优化，具体优化器类型在命令行参数中可选
        with tf.variable_scope("train_step"):
            self.global_step = tf.Variable(0, name="global_step", trainable=False)
            if self.optimizer == 'Adam':
                optim = tf.train.AdamOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'Adadelta':
                optim = tf.train.AdadeltaOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'Adagrad':
                optim = tf.train.AdagradOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'RMSProp':
                optim = tf.train.RMSPropOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'Momentum':
                optim = tf.train.MomentumOptimizer(learning_rate=self.lr_pl, momentum=0.9)
            elif self.optimizer == 'SGD':
                optim = tf.train.GradientDescentOptimizer(learning_rate=self.lr_pl)
            else:
                optim = tf.train.GradientDescentOptimizer(learning_rate=self.lr_pl)

            grads_and_vars = optim.compute_gradients(self.loss)#返回（梯度，变量）对的列表，梯度优化minimize的第一步骤
            #tf.clip_by_value(),截取g在【-self.clip_grad，self.clip_grad】之间
            grads_and_vars_clip = [[tf.clip_by_value(g, -self.clip_grad, self.clip_grad), v] for g, v in grads_and_vars]
            self.train_op = optim.apply_gradients(grads_and_vars_clip, global_step=self.global_step)#梯度优化minimize的第二部分，返回一个梯度执行更新的ops

    def init_op(self):#初始化
        self.init_op = tf.global_variables_initializer()

    def add_summary(self, sess):
        """

        :param sess:
        :return:
        """
        self.merged = tf.summary.merge_all()#将之前定义的所有summary op整合到一起
        self.file_writer = tf.summary.FileWriter(self.summary_path, sess.graph)#将摘要协议缓冲区写入事件文件

    def train(self, train, dev):
        """
        :param train:
        :param dev:
        :return:
        """
        saver = tf.train.Saver(tf.global_variables())#实例化saver

        with tf.Session(config=self.config) as sess:
            sess.run(self.init_op)#初始化变量
            self.add_summary(sess)#

            for epoch in range(self.epoch_num):
                self.run_one_epoch(sess, train, dev, self.tag2label, epoch, saver)

    def test(self, test):
        saver = tf.train.Saver()
        with tf.Session(config=self.config) as sess:
            self.logger.info('=========== testing ===========')
            saver.restore(sess, self.model_path)
            label_list, seq_len_list = self.dev_one_epoch(sess, test)
            self.evaluate(label_list, seq_len_list, test)

    def demo_one(self, sess, sent):
        """

        :param sess:
        :param sent:
        :return:
        """
        label_list = []
        for seqs, labels in batch_yield(sent, self.batch_size, self.vocab, self.tag2label, shuffle=False):
            label_list_, _ = self.predict_one_batch(sess, seqs)
            label_list.extend(label_list_)
        label2tag = {}
        for tag, label in self.tag2label.items():
            label2tag[label] = tag if label != 0 else label
        tag = [label2tag[label] for label in label_list[0]]
        return tag

    def run_one_epoch(self, sess, train, dev, tag2label, epoch, saver):
        """
        :param sess:
        :param train:
        :param dev:
        :param tag2label:
        :param epoch:
        :param saver:
        :return:
        """
        num_batches = (len(train) + self.batch_size - 1) // self.batch_size  #一次向神经网络喂入self.batch_size个数据，求出需要喂入数据的轮数，//表示整除
        ##batch_size为64
        start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())#时间戳
        #batches的类型是一个generator
        batches = batch_yield(train, self.batch_size, self.vocab, self.tag2label, shuffle=self.shuffle)


        for step, (seqs, labels) in enumerate(batches):#default：start=0，则step从0开始编号
            sys.stdout.write(' processing: {} batch / {} batches.'.format(step + 1, num_batches) + '\n')#输出信息:正在处理第1轮数据，共多少轮数据
            step_num = epoch * num_batches + step + 1#计步器   记录共运行了多少条数据
            feed_dict, seq_len_list = self.get_feed_dict(seqs, labels, self.lr, self.dropout_keep_prob)
            op, loss_train, summary, step_num_ = sess.run([self.train_op, self.loss, self.merged, self.global_step],
                                                         feed_dict=feed_dict)
            if step + 1 == 1 or (step + 1) % 300 == 0 or step + 1 == num_batches:#第1轮、每300轮、最后一轮 输出信息
                self.logger.info(
                    '{} epoch {}, step {}, loss: {:.4}, global_step: {}'.format(start_time, epoch + 1, step + 1,
                                                                                loss_train, step_num))

            self.file_writer.add_summary(summary, step_num)

            if step + 1 == num_batches:
                saver.save(sess, self.model_path, global_step=step_num)

        self.logger.info('===========validation / test===========')
        label_list_dev, seq_len_list_dev = self.dev_one_epoch(sess, dev)
        self.evaluate(label_list_dev, seq_len_list_dev, dev, epoch)

    def get_feed_dict(self, seqs, labels=None, lr=None, dropout=None):
        word_ids, seq_len_list = pad_sequences(seqs, pad_mark=0)

        feed_dict = {self.word_ids: word_ids,
                     # self.pos_ids:0,
                     self.sequence_lengths: seq_len_list}
        if labels is not None:
            labels_, _ = pad_sequences(labels, pad_mark=0)
            feed_dict[self.labels] = labels_
        if lr is not None:
            feed_dict[self.lr_pl] = lr
        if dropout is not None:
            feed_dict[self.dropout_pl] = dropout

        return feed_dict, seq_len_list

    def dev_one_epoch(self, sess, dev):
        """

        :param sess:
        :param dev:
        :return:
        """
        label_list, seq_len_list = [], []
        for seqs, labels in batch_yield(dev, self.batch_size, self.vocab, self.tag2label, shuffle=False):
            label_list_, seq_len_list_ = self.predict_one_batch(sess, seqs)
            label_list.extend(label_list_)
            seq_len_list.extend(seq_len_list_)
        return label_list, seq_len_list

    def predict_one_batch(self, sess, seqs):
        """
        :param sess:
        :param seqs:
        :return: label_list
                 seq_len_list
        """
        feed_dict, seq_len_list = self.get_feed_dict(seqs, dropout=1.0)

        if self.CRF:
            logits, transition_params = sess.run([self.logits, self.transition_params],
                                                 feed_dict=feed_dict)
            label_list = []
            for logit, seq_len in zip(logits, seq_len_list):
                viterbi_seq, _ = viterbi_decode(logit[:seq_len], transition_params)
                label_list.append(viterbi_seq)
            return label_list, seq_len_list

        else:
            label_list = sess.run(self.labels_softmax_, feed_dict=feed_dict)
            return label_list, seq_len_list

    def evaluate(self, label_list, seq_len_list, data, epoch=None):
        """
        :param label_list:
        :param seq_len_list:
        :param data:
        :param epoch:
        :return:
        """
        label2tag = {}
        #生成label和tag的对应关系，和tag2label相反，但是label为0的tag保留为0，其他不变
        for tag, label in self.tag2label.items():
            #label2tag[label] = tag if label != 0 else label
            label2tag[label] = tag

        model_predict = []
        for label_, (sent, tag) in zip(label_list, data):
            tag_ = [label2tag[label__] for label__ in label_]
            sent_res = []
            if  len(label_) != len(sent):
                print(sent)
                print(len(label_))
                print(tag)
            for i in range(len(sent)):
                sent_res.append([sent[i], tag[i], tag_[i]])
            model_predict.append(sent_res)
        epoch_num = str(epoch+1) if epoch != None else 'test'
        label_path = os.path.join(self.result_path, 'label_' + epoch_num)
        metric_path = os.path.join(self.result_path, 'result_metric_' + epoch_num)
        for _ in conlleval(model_predict, label_path, metric_path):
            self.logger.info(_)

