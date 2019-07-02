#!/usr/bin/env python
# coding=utf-8
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer

def countvec():
    # 实例化count
    count = CountVectorizer()
    # 对两篇文章特征抽取
    data = count.fit_transform(["life is short, i like python", "life is too long, i dislike python"])

    # 内容
    print(count.get_feature_names())
    print(data.toarray())
    print(data)

    return None

def dictvec():
    """对字典数据进行特征抽取"""
    # 实例化dictvec
    dic = DictVectorizer(sparse=False)
    # dictvec调用fit_transform
    # 三个样本的特征数据（字典形式）
    data = dic.fit_transform([{'city': '北京', 'temperature': 100},
                       {'city': '上海', 'temperature': 60},
                       {'city': '深圳', 'temperature': 30}])
    
    # 打印特征抽取后的特征结果
    print(dic.get_feature_names())
    print(data)
    return None


if __name__ == '__main__':
    # 默认返回sparse矩阵
    countvec()
    
