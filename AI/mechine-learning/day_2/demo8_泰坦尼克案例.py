#!/usr/bin/env python
# coding=utf-8
from sklearn.feature_extraction import DictVectorizer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz


def decision():
    """决策树预测乘客生存分类"""
    # 1.获取乘客的数据
    titan = pd.read_csv("http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt")
    # 2.确定特征值和目标值，缺失值处理，特征类别数据-->one-hot编码
    # 取出特征值
    x = titan[['pclass', 'age', 'sex']]
    # 取出目标值
    y = titan['survived']
    # 填充缺失值
    x['age'].fillna(x['age'].mean(), inplace=True)
    # 特征类别数据-->one-hot编码
    dic= DictVectorizer(sparse=False)
    # 转换示例：[["1st","2","female"],[]]--->[{"pclass":, "age":2, "sex: female"}, ]
    x = dic.fit_transform(x.to_dict(orient="records"))
    print(dic.get_feature_names())
    print(x)
    # 3、分割数据集
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

    # 4、决策树进行预测
    dec = DecisionTreeClassifier(max_depth=5)
    #
    dec.fit(x_train, y_train)
    #
    print("预测的准确率为：", dec.score(x_test, y_test))

    # 导出到dot文件
    export_graphviz(dec, out_file="./test.dot", feature_names=['age', 'pclass=1st', 'pclass=2nd', 'pclass=3rd', '女性', '男性'])


if __name__ == '__main__':
    decision()
