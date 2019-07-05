#!/usr/bin/env python
# coding=utf-8
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
import jieba
import pandas as pd 
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_selection import VarianceThreshold
from scipy.stats import pearsonr


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


def standardscaler():
    """对约会对象数据进行标准化处理"""
    # 读取数据选择要处理的特征
    dating = pd.read_csv("./dating.txt")
    data = dating[['milage', 'Liters', 'Consumtime']]
    # 实例化，进行fit_transform
    std = StandardScaler()
    data = std.fit_transform(data)
    print(data)
        
    return None 

def varthreshold():
    """使用方差法进行股票的指标的过滤"""
    factor = pd.read_csv("./stock_day.csv")
    # 使用VarianceThreshold
    var = VarianceThreshold(threshold=0.0)
    # 9列指标低方差过滤
    # data = var.fit_transform(factor.iloc[:, 1:10])
    data = var.fit_transform(factor)
    
    print(data)
    print(data.shape)
    return None

def pearson():
    """对股票的一些常见财务指标进行相关性计算"""
    factor = ['open', 'high', 'low', 'volume', 'price_change', 'p_change', 'ma5', 'ma10', 'ma20', 'v_ma5', 'v_ma10', 'v_ma20', 'turnover']
    data = pd.read_csv('./stock_day.csv')
    # 循环获取两个指标
    for i in range(len(factor)):
        for j in range(i, len(factor)-1):
            # 第一次open,j+1
            print("指标1:%s 和指标2:%s 的相关系数计算：%f" % (factor[i], factor[j+1], pearsonr(data[factor[i]], data[factor[j+1]])[0]))

    return None 



if __name__ == '__main__':
    pearson()



















