#!/usr/bin/env python
# coding=utf-8
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_20newsgroups
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, GridSearchCV
import pandas as pd 


def nbcls():
    """朴素贝叶斯对20类新闻进行分类"""
    news = fetch_20newsgroups(subset="all")
    # print(news)

    # 1.分割数据集
    x_train, x_test, y_train, y_test = train_test_split(news.data, news.target, test_size=0.3)
    
    # 2.进行文本特征抽取，让算法能够知道特征值
    # 实例化tfidf
    tfidf = TfidfVectorizer()
    # 对训练集特征抽取
    x_train = tfidf.fit_transform(x_train)
    # 对测试集特征抽取
    # 注意，此处不再是fit_transform，因我们是根据训练集统计出的特征来预测测试集
    x_test = tfidf.transform(x_test)

    # 进行朴素贝叶斯算法预测
    mlb = MultinomialNB(alpha=1.0)
    mlb.fit(x_train, y_train)
    # 预测，准确率
    print("预测测试集当中的文档类别是:", mlb.predict(x_test)[:50])
    print("测试集当中的文档真是类别是:", y_test[:50])

    # 得出准确率
    print("文档分类的准确率为:", mlb.score(x_test, y_test))


    return None 



if __name__ == '__main__':
    nbcls()
