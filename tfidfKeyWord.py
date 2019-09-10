#!/usr/bin/env python
# encoding: utf-8
'''
@author: fangbing
@contact: fangbing@cvte.com
@file: tfidfKeyWord.py
@time: 2018/9/11 8:41
@desc:
'''
import sys
import pandas as pd
import numpy as np
import jieba.posseg
import jieba.analyse
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import pickle

def dataPreppss(text,stopkey):
    l=[]
    pos=['n','nz','v','vd','vn','l','a','d']
    seg=jieba.posseg.cut(text)
    for i in seg:
        if i.word not in stopkey and i.flag in pos:
            l.append(i.word)
    return l

def getKeywords_tfidf(idList,titleList,contents,stopkey,topK):
    corpus=[]
    for text in contents:
        text=dataPreppss(text,stopkey)
        text=" ".join(text)
        corpus.append(text)
    print(len(corpus))
    vectorizer=CountVectorizer()
    X=vectorizer.fit_transform(corpus)
    tranforms=TfidfTransformer()
    tfidf=tranforms.fit_transform(X)
    word=vectorizer.get_feature_names()
    weight=tfidf.toarray()
    ids,titles,keys=[],[],[]
    for i in range(len(weight)):
        ids.append(idList[i])
        titles.append(titleList[i])
        df_word,df_weight=[],[]
        for j in range(len(word)):
            print(word[j],weight[i][j])
            df_word.append(word[j])
            df_weight.append(weight[i][j])
        df_word=pd.DataFrame(df_word,columns=['word'])
        df_weight=pd.DataFrame(df_weight,columns=['weight'])
        word_weight=pd.concat([df_word,df_weight],axis=1)
        word_weight=word_weight.sort_values(by='weight',ascending=False)
        keyword=np.array(word_weight['word'])
        word_split=[keyword[x] for x in range(0,topK)]
        word_split=" ".join(word_split)
        keys.append(word_split.encode('utf-8'))
    print(len(corpus))
    result=pd.DataFrame({"id":ids,"title":titles,"key":keys},columns=['id','title','key'])
    return result

def main():
    dataFile='./data/all_docs.txt'
    stop_words = pickle.load(open('./data/stop_word.pkl', 'rb'))
    f=open(dataFile,'r',encoding='utf-8')
    lines=f.readlines()
    ids=[]
    titles=[]
    contents=[]
    for line in lines:
        line=line.replace('\n','')
        id,title,content=line.split('\001')
        print(id,title,content)
        ids.append(id)
        titles.append(title)
        contents.append(content)
    result=getKeywords_tfidf(ids,titles,contents,stop_words,5)
    result.to_csv('./data/keys_tfidf.csv',index=False)
if __name__=='__main__':
    main()
