#!/usr/bin/env python
# coding=utf-8
from sklearn.datasets import load_iris, load_boston, fetch_20newsgroups

# 分类数据集
lr = load_iris()

# 回归数据集
lr2 = load_boston()

# print("特征值:", lr.data)
print("目标值(标签):", lr.target)
# print(lr.DESCR)
# print(lr.target_names)
# print(lr.feature_names)

print("特征值:", lr2.data)
print("目标值(标签数组):", lr2.target)
print(lr2.DESCR)
# print(lr2.target_names)
# print(lr2.feature_names)

# 数据量较大的数据集要用fetch下载
news = fetch_20newsgroups(subset='all')
print(news.data)
print(news.target)
