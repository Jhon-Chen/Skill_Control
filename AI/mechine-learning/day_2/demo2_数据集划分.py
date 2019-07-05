#!/usr/bin/env python
# coding=utf-8

from sklearn.datasets import load_iris, load_boston
from sklearn.model_selection import train_test_split

lr = load_iris()

# 进行数据集的训练集和测试集划分
# 返回值有四个部分接收
# x，y代表特征值目标值；train，test代表训练集和测试集
# 次序为：训练集的特征值，测试集的特征值，训练集标签值，测试集标签值
x_train, x_test, y_train, y_test = train_test_split(lr.data, lr.target, test_size=0.3)
print("训练集的特征值", x_train)
print("测试集的特征值", x_test)
print("训练集的标签值", y_train)
print("测试集的标签值", y_test)

