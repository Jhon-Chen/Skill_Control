#!/usr/bin/env python
# coding=utf-8
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
import jieba


def cutword():
    # 将文本用jieba处理
    content1 = jieba.cut("如梦令其一宋·李清照常记溪亭日暮，沉醉不知归路。兴尽晚回舟，误入藕花深处。争渡，争渡，惊起一滩鸥鹭。")
    content2 = jieba.cut("如梦令其二宋·李清照昨夜雨疏风骤，浓睡不消残酒。试问卷帘人，却道海棠依旧。知否？知否？应是绿肥红瘦")

    # 把分词后的对象转换成列表，再变成以空格隔开的字符串
    c1 = ' '.join(list(content1))
    c2 = ' '.join(list(content2))

    return c1, c2 



def tfidfvec():
    # 实例化count
    tfidf = TfidfVectorizer()

    # 进行分词操作
    c1, c2 = cutword()

    # 对文章进行特征抽取
    data = tfidf.fit_transform([c1, c2])

    # 内容
    print(tfidf.get_feature_names())
    print(data.toarray())


if __name__ == '__main__':
    tfidfvec()


















