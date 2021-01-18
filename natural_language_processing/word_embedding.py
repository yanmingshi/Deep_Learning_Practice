"""
_*_ coding: utf-8 _*_
@Time : 2020/11/2 10:02
@Author : yan_ming_shi
@Version：V 0.1
@File : word_embedding.py
@desc : 寻找语义相近的词
"""
from abc import ABC

import torch
import torch.nn as nn  # 神经网络工具箱torch.nn
import torch.nn.functional as F  # 神经网络函数torch.nn.functional
import torch.utils.data as tud  # Pytorch读取训练集需要用到torch.utils.data类

from torch.nn.parameter import Parameter  # 参数更新和优化函数

from collections import Counter  # Counter 计数器
import numpy as np
import random
import math

import pandas as pd
import scipy  # SciPy是基于NumPy开发的高级模块，它提供了许多数学算法和函数的实现
import sklearn
from sklearn.metrics.pairwise import cosine_similarity  # 余弦相似度函数

USE_CUDA = torch.cuda.is_available()  # 有GPU可以用

# 为了保证实验结果可以复现，我们经常会把各种random seed固定在某一个值
random.seed(53113)
np.random.seed(53113)
torch.manual_seed(53113)
if USE_CUDA:
    torch.cuda.manual_seed(53113)

# 设定一些超参数
K = 100  # number of negative samples 负样本随机采样数量
C = 3  # nearby words threshold 指定周围三个单词进行预测
NUM_EPOCHS = 2  # The number of epochs of training 迭代轮数
MAX_VOCAB_SIZE = 30000  # the vocabulary size 词汇表多大
BATCH_SIZE = 128  # the batch size 每轮迭代1个batch的数量
LEARNING_RATE = 0.2  # the initial learning rate #学习率
EMBEDDING_SIZE = 100  # 词向量维度

LOG_FILE = "word-embedding.log"


# tokenize函数，把一篇文本转化成一个个单词
def word_tokenize(text):
    return text.split()


with open("text8/text8.train.txt", "r") as file:
    text = file.read()

text = [w for w in word_tokenize(text.lower())]

vocab = dict(Counter(text).most_common(MAX_VOCAB_SIZE - 1))
# unk表示不常见单词数=总单词数-常见单词数
vocab['<unk>'] = len(text) - np.sum(list(vocab.values()))
# 取出字典的所有单词key
idx_to_word = [word for word in vocab.keys()]
# 取出所有单词的单词和对应的索引，索引值与单词出现次数相反，最常见单词索引为0。
word_to_idx = {word: i for i, word in enumerate(idx_to_word)}
# 所有单词的频数values
word_counts = np.array([count for count in vocab.values()], dtype=np.float32)
# 所有单词的频率
word_freqs = word_counts / np.sum(word_counts)
word_freqs = word_freqs ** (3. / 4.)
# 重新计算所有单词的频率
word_freqs = word_freqs / np.sum(word_freqs)  # 用来做 negative sampling
VOCAB_SIZE = len(idx_to_word)  # 词汇表单词数30000=MAX_VOCAB_SIZE


class WordEmbeddingDataset(tud.Dataset):

    def __init__(self, text, word_to_idx, idx_to_word, word_freqs, word_counts):
        """
            text: a list of words, all text from the training dataset
            word_to_idx: the dictionary from word to idx
            idx_to_word: idx to word mapping
            word_freq: the frequency of each word
            word_counts: the word counts
        """
        super(WordEmbeddingDataset, self).__init__()
        # 字典 get() 函数返回指定键的值（第一个参数），如果值不在字典中返回默认值（第二个参数）。
        # 取出text里每个单词word_to_idx字典里对应的索引,不在字典里返回"<unk>"的索引
        self.text_encoded = [word_to_idx.get(t, word_to_idx["<unk>"]) for t in text]
        self.text_encoded = torch.Tensor(self.text_encoded).long()
        self.word_to_idx = word_to_idx
        self.idx_to_word = idx_to_word
        self.word_freqs = torch.Tensor(word_freqs)
        self.word_counts = torch.Tensor(word_counts)

    def __len__(self):
        """
        返回整个数据集（所有单词）的长度
        :return:
        """
        return len(self.text_encoded)

    def __getitem__(self, idx):
        """
            返回以下数据用于训练
            - 中心词
            - 这个单词附近的(positive)单词
            - 随机采样的K个单词作为negative sample
        :param idx:单词编号
        :return:
        """
        center_word = self.text_encoded[idx]
        # 取中心词前后三个词
        pos_indices = list(range(idx - C, idx)) + list(range(idx + 1, idx + C + 1))
        pos_indices = [i % len(self.text_encoded) for i in pos_indices]  # 超出词汇总数时，取余数
        print(pos_indices)
        pos_words = self.text_encoded[pos_indices]

        neg_words = torch.multinomial(self.word_freqs, K * pos_words.shape[0], True)

        return center_word, pos_words, neg_words


dataset = WordEmbeddingDataset(text, word_to_idx, idx_to_word, word_freqs, word_counts)
dataloader = tud.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)


class EmbeddingModel(nn.Module):
    """
    embedding模型定义
    """

    def __init__(self, vocab_size, embed_size):
        super(EmbeddingModel, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size

        init_range = 0.5 / self.embed_size
        # 模型输出nn.Embedding(30000, 100)
        self.out_embed = nn.Embedding(self.vocab_size, self.embed_size, sparse=False)
        # 权重初始化
        self.out_embed.weight.data.uniform_(-init_range, init_range)

        # 模型输入nn.Embedding(30000, 100)
        self.in_embed = nn.Embedding(self.vocab_size, self.embed_size, sparse=False)
        # 权重初始化
        self.in_embed.weight.data.uniform_(-init_range, init_range)

    def forward(self, input_labels, pos_labels, neg_labels):
        """

        :param input_labels: 中心词, [batch_size]
        :param pos_labels: 中心词周围 context window 出现过的单词 [batch_size * (window_size * 2)]
        :param neg_labels: 中心词周围没有出现过的单词，从 negative sampling 得到 [batch_size, (window_size * 2 * K)]
        :return: loss, [batch_size]
        """
        batch_size = input_labels.size(0)
        input_embedding = self.in_embed(input_labels)  # B * embed_size
        pos_embedding = self.out_embed(pos_labels)  # B * (2*C) * embed_size
        neg_embedding = self.out_embed(neg_labels)  # B * (2*C * K) * embed_size

        # torch.bmm()为batch间的矩阵相乘（b,n.m)*(b,m,p)=(b,n,p) unsqueeze(2)指定位置升维
        log_pos = torch.bmm(pos_embedding, input_embedding.unsqueeze(2)).squeeze()  # B * (2*C)
        log_neg = torch.bmm(neg_embedding, -input_embedding.unsqueeze(2)).squeeze()  # B * (2*C*K)

        log_pos = F.logsigmoid(log_pos).sum(1)
        log_neg = F.logsigmoid(log_neg).sum(1)
        loss = log_pos + log_neg
        return -loss

    def input_embeddings(self):
        """
        取出self.in_embed数据参数
        :return:
        """
        return self.in_embed.weight.data.cpu().numpy()


if __name__ == '__main__':

    model = EmbeddingModel(VOCAB_SIZE, EMBEDDING_SIZE)
    if USE_CUDA:
        model = model.cuda()


    def evaluate(filename, embedding_weights):
        if filename.endswith(".csv"):
            data = pd.read_csv(filename, sep=",")
        else:
            data = pd.read_csv(filename, sep="\t")
        human_similarity = []
        model_similarity = []
        for i in data.iloc[:, 0:2].index:
            word1, word2 = data.iloc[i, 0], data.iloc[i, 1]
            if word1 not in word_to_idx or word2 not in word_to_idx:
                continue
            else:
                word1_idx, word2_idx = word_to_idx[word1], word_to_idx[word2]
                word1_embed, word2_embed = embedding_weights[[word1_idx]], embedding_weights[[word2_idx]]
                model_similarity.append(float(sklearn.metrics.pairwise.cosine_similarity(word1_embed, word2_embed)))
                human_similarity.append(float(data.iloc[i, 2]))

        return scipy.stats.spearmanr(human_similarity, model_similarity)  # , model_similarity


    def find_nearest(word):
        index = word_to_idx[word]
        embedding = embedding_weights[index]
        cos_dis = np.array([scipy.spatial.distance.cosine(e, embedding) for e in embedding_weights])
        return [idx_to_word[i] for i in cos_dis.argsort()[:10]]


    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    for e in range(NUM_EPOCHS):
        for i, (input_labels, pos_labels, neg_labels) in enumerate(dataloader):

            # TODO
            input_labels = input_labels.long()
            pos_labels = pos_labels.long()
            neg_labels = neg_labels.long()
            if USE_CUDA:
                input_labels = input_labels.cuda()
                pos_labels = pos_labels.cuda()
                neg_labels = neg_labels.cuda()

            optimizer.zero_grad()
            loss = model(input_labels, pos_labels, neg_labels).mean()
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                with open(LOG_FILE, "a") as fout:
                    fout.write("epoch: {}, iter: {}, loss: {}\n".format(e, i, loss.item()))
                    print("epoch: {}, iter: {}, loss: {}".format(e, i, loss.item()))

            if i % 2000 == 0:
                embedding_weights = model.input_embeddings()
                sim_simlex = evaluate("simlex-999.txt", embedding_weights)
                sim_men = evaluate("men.txt", embedding_weights)
                sim_353 = evaluate("wordsim353.csv", embedding_weights)
                with open(LOG_FILE, "a") as fout:
                    print(
                        "epoch: {}, iteration: {}, simlex-999: {}, men: {}, sim353: {}, nearest to monster: {}\n".format(
                            e, i, sim_simlex, sim_men, sim_353, find_nearest("monster")))
                    fout.write(
                        "epoch: {}, iteration: {}, simlex-999: {}, men: {}, sim353: {}, nearest to monster: {}\n".format(
                            e, i, sim_simlex, sim_men, sim_353, find_nearest("monster")))

        embedding_weights = model.input_embeddings()
        np.save("embedding-{}".format(EMBEDDING_SIZE), embedding_weights)
        torch.save(model.state_dict(), "embedding-{}.th".format(EMBEDDING_SIZE))
