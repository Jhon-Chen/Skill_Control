# 人工智能の机器学习_day1

[TOC]

## 概述

* 人工智能：50年代，自动化
* 机器学习：80年代，邮件分类
* 深度学习：最近十年，图像识别

### 机器学习定义

机器学习是从**数据**中心自**动分析获得规律**（模型），并利用规律对**未知数据进行预测**。



## 特征工程

### 什么是特征工程

特征工程是使用**专业的知识和技巧处理数据，使得特征能在机器学习算法上发挥更好的作用**的过程。
意义：会直接影响到算法的预测效果

### 为什么需要特征工程

筛选、处理选择合适的特征。

#### 数据集的构成

*可用数据集*

* scikit-learm：数据量较小、方便学习
* Kaggle：大数据竞赛平台、80万科学家、真实数据、数据量巨大
* UCI： 360个数据集、覆盖科学，生活，经济等领域、数据量几十万

*数据集的组成*

* 特征值
* 目标值

*特征工程包含内容*

* 特征抽取
* 特征预处理
* 特征降维

## Machine Learning with Scikit-Learn

### Scikit-learn包含的内容

scikit-learn接口：

* 分类、聚类、回归
* 特征工程
* 模型选择、调优

## 特征抽取

1. 包含将任意数据（如文本或图像，类别特征）转换为可用于机器学习的*数字特征*
   *注：特征值化是为了计算机更好的去理解数据*

   * 字典特征提取（特征离散化）
   * 文本特征提取
   * 图像特征提取（深度学习介绍）

2. 特征提取API

   ```python
   sklearn.feature_extraction
   ```

### 字典数据特征提取

*作用：对字典数据进行特征值化*

* sklearn.feature_extraction.DictVectorizer(sparse=True, ...)
  * DictVectorizer.fit_transform(X)：字典或者包含字典的迭代器返回值，默认返回sparse矩阵（节约存储空间）。
  * DictVectorizer.inverse_transform(X)：array数组或者sparse矩阵返回值，转换之前数据格式
  *  **DictVectorizer.get_feature_names()：返回类别名称**

**应用**

我们对以下数据进行特征提取：

```python
[{'city': '北京', 'temperature': 100},
{'city': '上海', 'temperature': 60},
{'city': '深圳', 'temperature': 30}]
```

*目的：对字典中有类别的数据做处理——one-hot 编码*

**流程分析**

* 实例化类 DicVectorizer
* 调用 fit_transform 方法输入数据并转换（注意返回格式）

```python
  1 #!/usr/bin/env python                       
  2 # coding=utf-8
  3 from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
  4 from sklearn.feature_extraction import DictVectorizer
  5 
  6 def countvec():
  7     # 实例化count
  8     count = CountVertorizer()
  9     data = count.fit_transform(["life is short, i like python", "life is too long, i dislike python"])
 10 
 11     # 内容
 12     print(count.get_feature_names())
 13     print(data.toarray)
 14     return None
 15 
 16 def dictvec():
 17     """对字典数据进行特征抽取"""
 18     # 实例化dictvec
 19     dic = DictVectorizer(sparse=False)
 20     # dictvec调用fit_transform
 21     # 三个样本的特征数据（字典形式）
 22     data = dic.fit_transform([{'city': '北京', 'temperature': 100},
 23                        {'city': '上海', 'temperature': 60},
 24                        {'city': '深圳', 'temperature': 30}])
 25 
 26     # 打印特征抽取后的特征结果
 27     print(dic.get_feature_names())
 28     print(data)
 29     return None
 30 
 31 
 32 if __name__ == '__main__':
 33     # 默认返回sparse矩阵
 34     dictvec()
```



### 文本的特征提取

*作用：对文本数据进行特征值化*

*  **sklearn.feature_extration.text.CountVectorizer(stop_words=[])**
  stop_words，停止词：这些词不能反映文章的主题，词语性质比较中性
  于是在统计时跳过
  
  * 返回词频矩阵（单词列表）
  
* CountVectorizer：

  * 单词列表 ：将文章里的所有单词统计到一个列表当中（重复的词只当做一次），默认会过滤掉单个字母
  * 对于单个字母对文章的主题没有影响，故忽略
  * 对每篇文章在词的列表中的出现次数统计结果
  * 对于中文来讲，也不统计单个汉字，还需要以逗号或者空格进行分词处理

  

* CountVectorizer.fit_transform(X)：文本或者包含文本字符串的可迭代对象，返回值：返回sparse矩阵

* CountVectorizer.inverse_transform(X)：array数组或者sparse矩阵，返回值：转换之前的数据格式

* CountVectorizer.get_feature_names()：返回值：单词列表

* **sklearn.feature_extraction.text.TfidVectorizer**

**流程分析**

* 实例化类CountVectorizer
* 调用fit_transform方法输入数据并转换（注意返回格式，利用toarray()进行sparse矩阵转换array数组）

```python
  1 #!/usr/bin/env python
  2 # coding=utf-8
  3 from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
  4 from sklearn.feature_extraction import DictVectorizer
  5 
  6 def countvec():
  7     # 实例化count
  8     count = CountVectorizer()                        
  9     # 对两篇文章特征抽取
 10     data = count.fit_transform(["life is short, i like python", "life is too long, i dislike python"])
 11 
 12     # 内容
 13     print(count.get_feature_names())
 14     print(data.toarray())
 15     print(data)
 16 
 17     return None
 18 
 19 def dictvec():
 20     """对字典数据进行特征抽取"""
 21     # 实例化dictvec
 22     dic = DictVectorizer(sparse=False)
 23     # dictvec调用fit_transform
 24     # 三个样本的特征数据（字典形式）
 25     data = dic.fit_transform([{'city': '北京', 'temperature': 100},
 26                        {'city': '上海', 'temperature': 60},
 27                        {'city': '深圳', 'temperature': 30}])
 28 
 29     # 打印特征抽取后的特征结果
 30     print(dic.get_feature_names())
 31     print(data)
 32     return None
```

### jieba分词处理

* jieba.cut()
  * 返回词语组成的生成器

需要安装jieba库

```python
pip install jieba
```

**案例分析**
对以下词进行特征值化：

```
如梦令（其一）
宋·李清照
常记溪亭日暮，沉醉不知归路。
兴尽晚回舟，误入藕花深处。
争渡，争渡，惊起一滩鸥鹭。
如梦令（其二）
宋·李清照
昨夜雨疏风骤，浓睡不消残酒。
试问卷帘人，却道海棠依旧。
知否？知否？应是绿肥红瘦。
```

* 分析
  * 准备句子，利用jieba.cut进行分词
  * 实例化CountVectorizer
  * 将分词结果变成字符串当做fit_transform的输入值

```python
  1 #!/usr/bin/env python
  2 # coding=utf-8
  3 from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
  4 from sklearn.feature_extraction import DictVectorizer
  5 import jieba
  6
  8 def cutword():
  9     # 将文本用jieba处理
 10     content1 = jieba.cut("如梦令其一宋·李清照常记溪亭日暮，沉醉不知归路。兴尽晚回舟，误入藕花深处。争渡，争渡，惊起一滩鸥鹭。")
 11     content2 = jieba.cut("如梦令其二宋·李清照昨夜雨疏风骤，浓睡不消残酒。试问卷帘人，却道海棠依旧。知否？知否？应是绿肥红瘦")
 12 
 13     # 把分词后的对象转换成列表，再变成以空格隔开的字符串
 14     c1 = ' '.join(list(content1))
 15     c2 = ' '.join(list(content2))                          
 17     return c1, c2
 20 
 21 def chinesevec():
 22     # 实例化count
 23     count = CountVectorizer()
 24 
 25     # 进行分词操作
 26     c1, c2 = cutword()
 27 
 28     # 对文章进行特征抽取
 29     data = count.fit_transform([c1, c2])
 30 
 31     # 内容
 32     print(count.get_feature_names())
 33     print(data.toarray())
 36 if __name__ == '__main__':
 37     chinesevec()

```

### Tf-idf文本特征提取

* 为了处理这种同一个词在很多片文章当中出现的次数都比较多的情况
* Tf-idf文本特征提取
  * 如果一个词或短语在一篇文章中出现的概率高，并且在其他文章中出现的概率低，就称这个词有很好的区分能力，适合用来分类。
  *  **TF-IDF作用：用以评估一个词对于一个文件集或一个语料库中的其中一份文件的重要程度。**

**公式**

* $tf\times idf = tfidf$
* 词频（term frequency，tf）指的是某一个给定的词语在该文件中出现的频率
* 逆向文档频率（inverse document frequency，idf）是一个词语普遍重要性的度量。某一特定词语的idf，可以由**总文件数目除以包含该词语的文件的数目，再讲得到的商取以10为底的对数得到**，最终得出的结果可以理解为重要程度。
* 对每一篇文章的重要性排序，找到每一篇文章的前N个关键词

*注：假如一篇文章的总词语数是100个，而词语“非常0出现了5次，那么“非常”一词在该文件中的词频就是5/100=0.05，计算文档频率（IDF）的方法是以文件集的文件总数，除以出现“非常”一词的文件数。所以，如果“非常”一词在1,000份文件中出现过，而文件的总数在10,000,000份的话，其逆向文件频率就是  $log(10,000,000/1,000) = 3$。最后“非常”对于这篇文章的 $tf-idf$的分数为$0.05\times3=0.15$*。

*重要性：Tf-idf分类机器学习算法进行文章分类中的前期数据处理方式*

## 特征预处理

### 什么是特征预处理

```python
# scikit-learn的解释
provides serveral common utility functions and transformer classes to change raw feature vectors into a representation that is more suitable for the downstream estimators.
```

即，通过一些**转换函数**将*特征数据*  **转换成更加适合算法模型**的*特征数据*过程。
可以借助下图理解：
![imagec86ea.png](https://miao.su/images/2019/07/02/imagec86ea.png)

特征预处理：基本的数据处理，比如缺失值处理（pandas）

* 数值型数据：通过转换函数进行特征数据的转换

### 包含内容

* 特征预处理API：

  ```python
  sklearn.preprocessing
  ```

* 数值型数据的无量纲化：
  * 归一化
  * 标准化
* 无量纲化的原因：
  特征的**单位**或**大小**相差较大，或者某特征的**方差**相比其他的特征要大出几个数量级，容易**影响（支配）目标结果**，使得一些算法无法学习到其他的特征。

### 无量纲化

![image4a339.png](https://miao.su/images/2019/07/02/image4a339.png)

对于以上的这种数据集我们就需要一些方法进行**无量纲化**，使**不同规格**的数据转换到**同一规格**。

### 归一化

定义：通过对原始数据进行变换把数据映射到（默认为[0,1])之间

公式：
$$
X_1 = \frac{x-min}{max-min}
$$

$$
X_2 = X_1\times(mx-mi)+mi
$$



针对以下数据：
![imagec86ea.png](https://miao.su/images/2019/07/02/imagec86ea.png)

> 作用于每一列，$x$为当前值，$max$为每一列的最大值，$min$为每一列的最小值，那么$X_2$为最终结果，$mx$、$mi$为指定需要转换的区间范围，默认$mx$为1，$mi$为0。

故：
$$
特征1.1：X_1=\frac{90-60}{90-60}=1,X_2=1\times(1-0)+0=1
$$

$$
特征2.1：X_1=\frac{2-2}{4-2}=0,X_2=0，其余省略
$$

![image4ad3e.png](https://miao.su/images/2019/07/02/image4ad3e.png)

#### API

* sklearn.preprocessing.MinMaxScaler(feature_range=(0,1)...)
  * MinMaxScaler.fit_transform(X)
    * X： numpy array格式的数据[n_sample, n_features]
  * 返回值：转换后的形状相同的array

#### 数据计算

我们对以下数据进行计算，数据保存在dating.txt中：

分析：

1. 导入模块`from sklearn.processing import MinMaxScaler`

2. 实例化`MinMaxScaler`

3. 通过`fit_transform转换`

   ```python
    38 def minmaxscaler():
    39     """对约会对象数据进行归一化处理"""
    40     # 读取数据选择要处理的特征
    41     dating = pd.read_csv("./dating.txt")
    42     data = dating[['milage', 'Liters', 'Consumtime']]
    43     # 实例化，进行fit_transform
    44     mm = MinMaxScaler(feature_range=(0, 9))
    45     data = mm.fit_transform(data)
    46     print(data)
    47         
    48     return None 
   
    54 if __name__ == '__main__':
    55     minmaxscaler()
   ```

**问题：**如果数据中的异常点较多，会有什么影响？

![image907dc.png](https://miao.su/images/2019/07/02/image907dc.png)

*注意，最大值和最小值是变化的，另外，最大值与最小值非常容易受到异常点影响，所以这种方法鲁莽性较差，只适合传统精确小数据场景。*

### 标准化

定义：通过对原始数据进行变换把数据变换到均值为0，标准差为1范围内

公式：
$$
X=\frac{x-mean}{\delta}
$$

> 作用于每一列，$mean$为平均值，$\delta$为标准差

所以回到刚才异常点的地方，我们再来看看标准化

![imagea65a1.png](https://miao.su/images/2019/07/02/imagea65a1.png)

* 对于归一化来说：如果出现异常点，影响了最大和最小值，那么结果显然会发生改变
* 对于标准化来说：如果出现异常点，由于具有**一定的数据量**，少量的异常点对于平均值的影响不大，从而方差的变化较小

#### API

* sklearn.preprocessing.StandardScaler()
  * 处理之后**每列来说所有的数据都聚集在**均值0**附近**标准差为1**的范围
  * StandardScaler.fit_transform(X)
    * X： numpy array 格式的数据[n_sample，n_features]
  * 返回值：转换后的形状相同的array

*数据处理：*

```python
    form sklearn.preprocessing import StandardScaler
 38 def standardscaler():
 39     """对约会对象数据进行标准化处理"""
 40     # 读取数据选择要处理的特征
 41     dating = pd.read_csv("./dating.txt")
 42     data = dating[['milage', 'Liters', 'Consumtime']]
 43     # 实例化，进行fit_transform
 44     std = StandardScaler()
 45     data = std.fit_transform(data)
 46     print(data)    
 48     return None 
 49     
 51 if __name__ == '__main__':
 52     standardscaler()
```

![image5e2ec.png](https://miao.su/images/2019/07/02/image5e2ec.png)

*总结：在已有样本足够多的的情况下比较稳定，适合现代嘈杂大数据的场景。
一般也只选用标准化做无量纲操作。*

##  特征选择

### 降维

降维是指在某些限定条件下。**降低随机变量（特征）个数**，得到一组**不相关**主变量的过程

> 正式因为在进行训练的时候，我们都是使用特征进行学习。如果特征本身存在问题或者特征之间相关性较强，对于算法学习预测会影响较大

* 维度：特征的数量
* 相关特征（correlate feature）
  * 类似相对湿度与降雨量
* 降维：去除相关特征

**降维的两种方式**

* 特征选择
* 主成分分析（可以理解一种特征提取的方式）

### 什么是特征选择

**定义**

数据中心包含**冗余或无关变量（或称特征、属性、指标等），**旨在**从原有特征中找出主要特征**。
![image48997.png](https://miao.su/images/2019/07/03/image48997.png)

**方法**

* Filter（过滤式）：主要探究特征本身特点、特征与特征的目标值之间关联
  * 方差选择器：低方差特征过滤
    * 方差很小：所有样本的某个特征值基本上一样
    * 方差很大：所有样本的某个特征值的差异性较大
  * 相关系数
* Embedded（嵌入式）：算法自动选择特征（特征与目标值之间的关联）
  * 决策树：信息熵、信息增益
  * 正则化：L1、L2
  * 深度学习：卷积等
* Wrapper（包裹式）

> 对于Embedded方式，只能在讲解算法的时候再进行介绍

**模块**

`sklearn.feature_selection`

### 过滤式

**低方差特征过滤**

删除低方差的一些特征，前面讲过方差的意义。再结合方差的大小来考虑这个方式的角度。

* 特征方差小：某个特征大多样本的值比较接近
* 特征方差大：某个特征很多样本的值都有差别
* 不太好选择这个方差的值，作为明显一些特征的处理

**API**

* `sklearn.feature_selection.VarianceThreshold(threshold=0.0)`
  * 删除所有低方差特征
  * `Variance.fit_transform(X)`
    * X：`numpy array`格式的数据[n_samples, n_features]
    * 返回值：训练集差异低于`threshold`的特征将被删除。默认值是保留所有非零方差特征，即删除所有样本中具有相同值的特征。

**数据计算**

我们对**某些股票的指标特征之间进行一个筛选**，数据在`factor_regression_data/factor_returns.csv`文档当中，除去`index, data, return`列不考虑（**这些类型不匹配，也不是所需要指标**）

```python
  8 from sklearn.feature_selection import VarianceThreshold
 51 def varthreshold():
 52     """使用方差法进行股票的指标的过滤"""
 53     factor = pd.read_csv("./stock_day.csv")
 54     # 使用VarianceThreshold
 55     var = VarianceThreshold(threshold=0.0)
 56     # 9列指标低方差过滤
 57     # data = var.fit_transform(factor.iloc[:, 1:10]) 
 58     data = var.fit_transform(factor)
 59 
 60     print(data)
 61     print(data.shape)
 62     return None
 63 
 65 if __name__ == '__main__':
 66     varthreshold()
```

### 相关系数

* 皮尔逊相关系数（Pearson Correlation Coefficient）
  * 反映变量之间相关关系密切程度的统计指标
* 公式计算案例：（n为样本数量）
  ![imagec8cfe.png](https://miao.su/images/2019/07/03/imagec8cfe.png)

*特点：*
相关系数的值介于$[-1,1]$，即$-1\leq r \leq 1$其性质如下：

* 当$r > 0$时，表示两变量正相关，$r<0$时，两变量负相关
* 当$|r|=1$时，表示两变量完全相关，当$r=0$时，表示两变量无相关关系
* 当$0\leq |r| \leq1$时，表示两变量存在一定程度的相关。且$|r|$越接近于1，两变量间线性关系越密切；$|r|$越接近于0，表示两变量线性相关越弱
* 一般可安三级划分：$|r|<0.4$为低相关度；$0.4\leq |r| \leq0.7$为显著性相关；$0.7\leq|r|\leq1$为高度线性相关

**API**

* `from scripy.stats import pearsonr`
  * `x:(N,) array_like`
  * `y:(N,) array_like Returns: (Pearson's correlation coefficient, p-value)`

### 案例：股票的财务指标相关性计算

```python
  9 from scipy.stats import pearsonr
    
 65 def pearson():
 66     """对股票的一些常见财务指标进行相关性计算"""
 67     factor = ['open', 'high', 'low', 'volume', 'price_change', 'p_change', 'ma5', 'ma10', 'ma20', 'v_ma5', 'v_ma10', 'v_ma20', 'turnover']               
 68     data = pd.read_csv('./stock_day.csv')
 69     # 循环获取两个指标
 70     for i in range(len(factor)):
 71         for j in range(i, len(factor)-1):
 72             # 第一次open,j+1
 73             print("指标1:%s 和指标2:%s 的相关系数计算：%f" % (factor[i], factor[j+1], pearsonr(data[factor[i]], data[factor[j+1]])[0]))
 74 
 75     return None
 76 
 79 if __name__ == '__main__':
 80     pearson()
```

可以利用图像来观察某两个指标的相关状况：
[![imagead1f8.png](https://miao.su/images/2019/07/03/imagead1f8.png)](https://miao.su/image/TDzLj)

所以相关的特征必须做一些相应的处理（删掉一种一些；合成一个新的）

### 主成分分析

什么是主成分分析（PCA）

* 定义：**高维数据转化为低维数据的过程**，在此过程中可能会**舍弃原有数据、创造新的变量**
* 作用：**是数据维数压缩 ，尽可能降低原数据的维数（复杂度），损失少量信息。**
* 应用：回归分析或者聚类分析当中
* 场景：
  * 特征数量非常大的时候：PCA去压缩掉相关的、冗余的信息，放置高维度时计算量过大
  * 创造新的变量（特征）：把相关系数高的压缩成一个新的特征

可以借助下图理解这个过程：
![image03f63.png](https://miao.su/images/2019/07/03/image03f63.png)

**API**

* `sklearn.decomposition.PCA(n_components=None)`
  * 将数据分解为较低维数空间
  * `n_components:`
    * *小数：表示保留百分之多少的信息*（90%以上）
    * *整数：减少到多少特征*
  * `PCA.fit_transform(X):numpy array 格式的数据[n_samples, n_features]`
  * 返回值：转换后制定维度的`array`

**数据计算**

先拿个简单的数据计算一下
![imageaad57.png](https://miao.su/images/2019/07/03/imageaad57.png)

```python
 80 def pc_a():
 81     """主成分分析进行降维"""
 82     data = pd.read_csv('./stock_day.csv')
 83     pca = PCA(n_components=0.95)
 84     data = pca.fit_transform(data[['high', 'open']])
 85     print(data)
 86 
 87     return None
 88 
 89 if __name__ == '__main__':
 90     pc_a()
```

### 案例：探究用户对物品类别的喜好细分降维

![image55045.png](https://miao.su/images/2019/07/03/image55045.png)

数据如下：

* `order_products_prior.csv`：订单与商品信息
  * 字段：**`order_id, product_id, add_to_cart_order, reordered`**
* `products.csv`：商品信息
  * 字段：`product_id, product_name, aisle_id, department_id`
* `orders.csv`：用户的订单信息
  * 字段：`aisle_id, aisle`

**需求**
将134个特征转变为44个特征。

![imaged6b48.png](https://miao.su/images/2019/07/03/imaged6b48.png)

**分析**

* 合并表，使得**user_id**与**aisle**在一张表中









## 机器学习算法介绍

### 机器学习类别

按照学习方式分类：

* **监督学习（supervised learning）（预测）**
  * 定义：输入数据是由输入特征值和目标值所组成。函数的输出可以是一个连续的值（称为回归），或是输出是有限个离散值（称作分类）。
  * **分类：k-近邻算法、贝叶斯分类、决策树与随机森林、逻辑回归、神经网络**
  * **回归：线性回归、岭回归**
  * 标注：隐马尔可夫模型（暂不要求？）
* **无监督学习（unsupervised learning）**
  * 定义：输入数据是由输入特征值所组成
  * **聚类：k-means**
* 半监督和强化学习

区别：
![imageabc53.png](https://miao.su/images/2019/07/03/imageabc53.png)

两种数据类型：

* 离散型数据：**由记录不同类别个体的数目所得到的数据，又称计数数据，所有这些数据全部都是整数，而且不能在细分，**也不能进一步提高他们的精度
* 连续型数据：**变量可以在某个范围内取任一数，即变量的取值可以是连续的，如，长度、时间、质量值等，**这类整数通常是非整数，含有小数部分。

> 注：离散型是区间不可分，连续型是区间内可分

* 分类：目标值数据类型为离散型
  * 分类是监督学习的一个核心问题，在监督学习中，当输出变量取有限个离散值时，预测问题变成为分类问题，**最基础的便是二分类问题，即判断是非，**从两个类别中选择一个作为预测结果；
* 回归：目标数据类型为连续型
  * 回归是监督学习的另一个重要问题。**回归用于预测输入变量和输出变量之间的关系，输出是连续型的值。**

### 机器学习开发流程

模型：指对于某个实际问题或客观事物，应用合适的算法得到的结果，称之为模型。

>day_1 end

