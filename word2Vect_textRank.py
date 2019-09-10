#!/usr/bin/env python
# encoding: utf-8
'''
@author: fangbing
@contact: fangbing@cvte.com
@file: word2Vect_textRank.py
@time: 2018/9/29 10:33
@desc:
'''
from nltk.tokenize import sent_tokenize,word_tokenize
from nltk.corpus import stopwords
import math
from itertools import product,count
from string import punctuation
from heapq import nlargest
import numpy as np
import jieba
from gensim import corpora, models,similarities
from gensim.models import Word2Vec
import pickle
import jieba.analyse


model=''
def prepareData():
    dataFile = 'data/all_docs.txt'
    f = open(dataFile, 'r', encoding='utf-8')
    lines = f.readlines()
    ids = []
    titles = []
    contents = []
    stopwords=pickle.load(open('data/stop_word.pkl','rb'))
    # jieba.analyse.set_stop_words('data/stopwords.txt')
    trainData=[]
    for line in lines:
        line = line.replace('\n', '')
        id, title, content = line.split('\001')
        ids.append(id)
        titles.append(title)
        contents.append(content)

        wordList = jieba.cut(titles+contents)
        new_wordList = []
        for word in wordList:
            if word not in stopwords:
                new_wordList.append(word)
        trainData.append(new_wordList)
    print(len(ids))
    trainWord2Vec(trainData)


def trainWord2Vec(traindata):
    dictionary = corpora.Dictionary(traindata)  ##得到词典
    token2id = dictionary.token2id
    corpus = [dictionary.doc2bow(text) for text in traindata]  ##统计每篇文章中每个词出现的次数:[(词编号id,次数number)]
    print('dictionary prepared!')
    ##接下来四行得到lda向量；
    tfidf = models.TfidfModel(corpus=corpus, dictionary=dictionary)
    wdfs = tfidf.dfs
    model = Word2Vec(traindata, size=200, window=5, min_count=1, workers=4)
    Word2Vec.load()
    model.save('./data/word2vec')
    print('word2vec model finish!')


def cut_sentences(sentence):
    puns = frozenset(u'。！？')
    tmp = []
    for ch in sentence:
        tmp.append(ch)
        if puns.__contains__(ch):
            yield ''.join(tmp)
            tmp = []
    yield ''.join(tmp)



def filter_symbols(sents):
    stopwords = create_stopwords() + ['。', ' ', '.']
    _sents = []
    for sentence in sents:
        for word in sentence:
            if word in stopwords:
                sentence.remove(word)
            if sentence:
                _sents.append(sentence)
    return _sents

def filter_model(sents):
    _sents = []
    for sentence in sents:
        for word in sentence:
            if word not in model:
                sentence.remove(word)
        if sentence:
            _sents.append(sentence)
    return _sents

def weight_sentences_rank(weight_graph):
    '''
    输入相似度的图（矩阵)
    返回各个句子的分数
    :param weight_graph:
    :return:
    '''
    # 初始分数设置为0.5
    scores = [0.5 for _ in range(len(weight_graph))]
    old_scores = [0.0 for _ in range(len(weight_graph))]
    # 开始迭代
    while different(scores, old_scores):
        for i in range(len(weight_graph)):
            old_scores[i] = scores[i]
        for i in range(len(weight_graph)):
            scores[i] = calculate_score(weight_graph, scores, i)
    return scores


def summarize(text, n):
    tokens = cut_sentences(text)
    sentences = []
    sents = []
    for sent in tokens:
        sentences.append(sent)
        sents.append(cutSentence(sent))
        # sents = filter_symbols(sents)
    sents = filter_model(sents)
    graph = create_graph(sents)
    scores = weight_sentences_rank(graph)
    print(scores)
    #zip(scores, count())
    sent_selected = nlargest(n, zip(scores,range(len(scores))))
    print(sent_selected)
    sent_index = []
    for i in range(n):
        if(i<len(sent_selected)):
            sent_index.append(sent_selected[i][1])
    print(sent_index)
    return [sentences[i] for i in sent_index]

stopwords=pickle.load(open('data/stop_word.pkl','rb'))

def cutSentence(sent):
    wordList = jieba.cut(sent)
    new_wordList = []
    for word in wordList:
        if word not in stopwords:
            new_wordList.append(word)
    return new_wordList

# 句子中的stopwords
def create_stopwords():
    stop_list = [line.strip() for line in open("stopwords.txt", 'r', encoding='utf-8').readlines()]
    return stop_list


"""传入两个句子返回这两个句子的相似度"""
def calculate_similarity(sen1, sen2):
    # 设置counter计数器
    counter = 0
    for word in sen1:
        if word in sen2:
            counter += 1
    return counter / (math.log(len(sen1)) + math.log(len(sen2)))

"""
传入句子的列表
返回各个句子之间相似度的图
（邻接矩阵表示）
"""
def create_graph(word_sent):
    num = len(word_sent)
    # 初始化表
    board = [[0.0 for _ in range(num)] for _ in range(num)]
    for i, j in product(range(num), repeat=2):
        if i != j:
            board[i][j] = compute_similarity_by_avg(word_sent[i], word_sent[j])
    return board

"""输入相似度邻接矩阵返回各个句子的分数"""
def weighted_pagerank(weight_graph):
    # 把初始的分数值设置为0.5
    scores = [0.5 for _ in range(len(weight_graph))]
    old_scores = [0.0 for _ in range(len(weight_graph))]
    # 开始迭代
    while different(scores, old_scores):
        for i in range(len(weight_graph)):
            old_scores[i] = scores[i]
        for i in range(len(weight_graph)):
            scores[i] = calculate_score(weight_graph, scores, i)
    return scores

"""判断前后分数有没有变化这里认为前后差距小于0.0001分数就趋于稳定"""
def different(scores, old_scores):
    flag = False
    for i in range(len(scores)):
        if math.fabs(scores[i] - old_scores[i]) >= 0.0001:
            flag = True
            break
    return flag

"""根据公式求出指定句子的分数"""
def calculate_score(weight_graph, scores, i):
    length = len(weight_graph)
    d = 0.85
    added_score = 0.0
    for j in range(length):
        fraction = 0.0
        denominator = 0.0
        # 先计算分子
        fraction = weight_graph[j][i] * scores[j]
        # 计算分母
        for k in range(length):
            denominator += weight_graph[j][k]
            if denominator == 0:
                denominator = 1
        added_score += fraction / denominator
    # 算出最终的分数
    weighted_score = (1 - d) + d * added_score
    return weighted_score


def cosine_similarity(vec1, vec2):
    '''
    计算两个向量之间的余弦相似度
    :param vec1:
    :param vec2:
    :return:
    '''
    tx = np.array(vec1)
    ty = np.array(vec2)
    cos1 = np.sum(tx * ty)
    cos21 = np.sqrt(sum(tx ** 2))
    cos22 = np.sqrt(sum(ty ** 2))
    cosine_value = cos1 / float(cos21 * cos22)
    return cosine_value

def compute_similarity_by_avg(sents_1, sents_2):
    '''    对两个句子求平均词向量    :param sents_1:    :param sents_2:    :return:    '''
    if len(sents_1) == 0 or len(sents_2) == 0:
        return 0.0
    index=0
    try:
      vec1 = model[sents_1[index]]
      vec2 = model[sents_2[index]]
    except:
      index=1
      vec1 = model[sents_1[index]]
      vec2 = model[sents_2[index]]

    for word1 in sents_1[index+1:]:
        vec1 = vec1 + model[word1]
    #vec2 = model[sents_2[index]]
    for word2 in sents_2[index+1:]:
        vec2 = vec2 + model[word2]
    similarity = cosine_similarity(vec1 / len(sents_1), vec2 / len(sents_2))
    return similarity


if __name__ == '__main__':
    # prepareData()
    # 记载已经训练好的中文模型
    f = open('simSentences.txt', 'a', encoding='utf-8')
    model = Word2Vec.load("./data/word2vec")
    count=0
    with open("./data/all_docs.txt", "r", encoding='utf-8') as myfile:
        while(myfile.readline() is not None):
            count+=1
            if(count<83304):
               continue
            print('count==',count)
            text = myfile.readline().replace('\n', '')
            print(text)
            id, title, content = text.split('\001')
            result=summarize(content, 2)
            print(result)
            if(len(result)==1):
                f.write(id + '\t' + result[0] + '\n')
            elif(len(result)==2):
                f.write(id+'\t'+result[0]+'\t'+result[1]+'\n')
            else:
                continue
    f.close()
