# 人工智能の机器学习day_2

[TOC]

## 数据集介绍与划分

### 数据集的划分

机器学习一般的数据集会划分成两个部分：

* 训练集：用于训练、构建模型
* 测试集：在模型检验时使用，用于评估模型是否有效

划分比例：

* 训练集：70%、80%、75%
* 测试集：30%、20%、25%

### API

* `sklearn.model_selection.train_test_split(arrarys, *options)`
  * x 数据集的特征值
  * y 数据集的目标值（标签）
  * test_size 测试集的大小，一般为float
  * random_state 随机数种子，不同的种子会造成不同的随机采样结果。相同的种子采样结果相同。
  * return，训练集特征值，测试集特征值，训练标签，测试标签（默认值随机取）

## sklearn数据集介绍

### API

* `sklearn.datasets`
  * 加载获取流行数据集
  * `datasets.load_*()`
    * 获取小规模数据集 `*` ，数据包含在`datasets`里
  * `datasets.fetch_*(data_home=None)`
    * 获取大规模数据集`*`，需要从网络上下载，函数的第一个参数是`data_home`，表示数据集下载的目录，默认是`~/scikit_learn_data/`

### 分类和回归数据集

* **分类数据集**
  `sklearn.datasets.load_iris()`加载并返回鸢尾花数据集
  `sklearn.datasets.load_digits()`加载并返回数字数据集

  * `sklearn.datasets.fetch_20newsgroups(data_home=None, subset='train')`
    *   subset：`train`或者`test`，`all`，可选，选择要加载的数据集。即训练集的“训练”，测试集的“测试”，两者的“全部”。

* **回归数据集**

  `sklearn.datasets.load_boston()`加载并返回波士顿房价数据集

  `sklearn.datasets.load_diabetes()`加载并返回糖尿病数据集

### 返回类型

* `load`和`fetch`返回的数据类型`datasets.base.Bunch(字典格式)`
  * `data`：特征数据数组，是`[n_samples * n_features]`的二维`numpy.ndarray`数组
  * `target`：标签数组，是`n_samples`的以为`numpy.ndarray`数组
  * `DESCR`：数据描述
  * `feature_names`：特征名，新闻数据，手写数组、回归数据集
  * `target_names`：标签名

## sklearn的转换器和估计器

### 转换器

想一下之前做的特征工程的步骤：

* 1.实例化（实例化一个转换器类（Transformer））
* 2.调用fit_transform（对于文档建立分类词频矩阵，不能同时调用）

我们把特征工程的接口称之为转换器，其中转换器调用有这么几种形式（以标准化为例）

* fit_transform(a)：相当于 fit + transform 以a自己的平均值标准差转换自己
* fit(a)：计算a的平均值标准差
* transform(b)：用a的平均值标准差去转换b

这个法则适用于特征选择、主成分分析、特征抽取 

可以通过ipython来观察他们之间的区别：
![imageaa509.png](https://miao.su/images/2019/07/04/imageaa509.png)

### 估计器（estimator）

*sklearn 机器学习算法的实现*
在sklearn中，估计器（estimator）是一个重要角色，是一类实现了算法的API。

1. 用于分类的估计器：
   * `sklearn.neighbors k`——近邻算法
   * `sklearn.naive_bayes`——贝叶斯
   * `sklearn.linear_model.LogisticRegression`——逻辑回归
   * `sklearn.tree`——决策树与随机森林
2. 用于回归的估计器：
   * `sklearn.linear_model.LinearRegression`——线性回归
   * `sklearn.linear_model.Ridge`——岭回归
3. 用于无监督学习的估计器
   * `sklearn.cluster.KMeans`——聚类

### 估计器工作流程

![image02e37.png](https://miao.su/images/2019/07/04/image02e37.png)

* fit()：训练集训练
* 测试集预测：predict、score

![image6b8d9.png](https://miao.su/images/2019/07/04/image6b8d9.png)



## K-近邻算法（KNN）

### 定义

如果一个样本在特征空间中的**k个最相似（即特征空间中最邻近）的样本中的大多数属于某一个类别**，则该样本也属于这个类别。

> 来源：KNN算法最早是由Cover和Hart提出的一种分类算法

### 距离公式

两个样本的距离可以通过如下公式计算，也称欧式距离（空间两点的距离公式）
	$点x(a_1,b_1,c_1)和点y(a_2,b_2,c_2)的距离公式：$
$$
\sqrt[]{(a_1-a_2)^2 + (b_1-b_2)^2 + (c_1-c_2)^2}
$$

### 电影类型分析

假设我们有限制几部电影：
![imageb34be.png](https://miao.su/images/2019/07/04/imageb34be.png)
其中`?`电影不知道类别，如何去预测？我们可以利用k-近邻算法的思想。
![image78963.png](https://miao.su/images/2019/07/04/image78963.png)

### 问题

* 如果距离最近的电影数量不止一个？

### K-近邻算法API

* `sklearn.neighbors.KNeighborsClassifier(n_neighbours=5, algorithm='auto')`
  * `n_neighbors:int,可选（默认=5）, k_neighbors查询默认使用的邻居数`
  * `algorithm:{'auto','ball_tree','kd_tree','brute'}`，可选用于计算最近邻居的算法：`ball_tree`将会使用`BallTree`，`kd_tree`将使用`KDTree`。`auto`将尝试根据传递给`fit`方法的值来决定最合适的算法。（不同的实现方法影响效率）。

### 案例：预测签到位置

![image252ac.png](https://miao.su/images/2019/07/04/image252ac.png)

数据介绍：

```python
train.csv, test.csv
row_id: 登记事件的ID
xy: 坐标
准确性: 定位准确性
时间: 时间戳
place_id: 业务的ID，这是你要预测的目标
```



#### 分析

* 对于数据做一些一些基本处理（这里所做的一些处理不一定达到很好的效果，我们只是简单的尝试，有些特征我们可以根据一些特征选择的方式去处理）

  1. 缩小数据集范围	DataFrame.query()

  2. 删除没用的日期数据     DataFrame.drop（可以选择保留）

  3. 将签到位置少于n个用户的信息删除

     ```python
     place_count = data.groupby('place_id').count()
     tf = place_count[place_count.row_id > 3].reset_index()
     data = data[data['place_id'].isoin(tf.place_id)]
     ```

* 分割数据集

* 标准化处理

* k-近邻预测

**K-近邻算法的一般流程**

1. 收集数据：可以使用任何方法
2. 准备数据：距离计算所需要的数值，最好是结构化的数据格式
3. 分析数据：可以使用任何方法
4. 训练算法：此步骤不适合于K-近邻算法
5. 测试算法：计算错误率
6. 使用算法：首选需要输入样本数据和结构化的输出结果，然后运行K-近邻算法判定输入数据分别属于哪个分类，最后应用对计算出的分类执行后续的处理。

#### 代码

```python
  8 def knncls():
  9     """K-近邻算法预测用户查询业务"""
 10     data = pd.read_csv('/home/jhonchen/scikit_learn_data/FBlocation/train_0.csv')
 11     # print(data)
 12     # 1.缩小数据范围
 13     # data = data.query("x >1.0 & x < 1.25 & y > 2.5 & y< 2.75")
 14     # 取出特征值和目标值
 15     y = data[['place_id']]
 16 
 17     x = data[['x', 'y', 'accuracy', 'time']]
 18     x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
 19     # 进行数据的标准化处理
 20     # 实例化
 21     std = StandardScaler()
 22     # 对训练集的特征值做标准化处理
 23     x_train = std.fit_transform(x_train)
 24     # 对测试集的特征值做标准化处理
 25     x_test = std.fit_transform(x_test)
 26 
 29     # 5.利用k-近邻算法去进行训练预测
 30     knn = KNeighborsClassifier(n_neighbors=5)
 31     # 调用fit和predict或者score
 32     knn.fit(x_train, y_train)
 33     # 预测测试集的目标值（签到位置）
 34     y_predict = knn.predict(x_test)
 35 
 36     print("K近邻算法预测的这些时间的业务类型：", y_predict)
 37     print("K近邻预测的准确率：", knn.score(x_test, y_test))
 38     return None
 
 41 if __name__ == '__main__':
 42     knncls()
```

#### 结果分析

**准确率：分类算法的评估之一**

1. k值取多大？有什么影响？
   k值取的很小：容易受到异常点的影响
   k值取的很大：会受到样本均衡的问题
2. 性能问题？
   距离计算上面，时间复杂度高

### 总结

* 优点：
  简单、易于理解、易于实现、无需训练
* 缺点：
  懒惰算法、对测试样本分类时的计算量大、内存开销大
  必须指定k值，k值选择不当则分类精度不能保证
* 使用场景：
  小数据场景，几千~几千万样本，具体场景具体业务去测试



## 模型的选择与调优

### 为什么要交叉验证

交叉验证的目的：为了让被评估的模型更加准确可信，为了看一个参数在这个数据集当中综合表现情况

### 什么是交叉验证(cross validation)

交叉验证：将拿到的**训练数据**(和测试集无关)，分为**训练集**和**验证集**。以下图为例：将数据分为5份，其中一份作为验证集。然后经过5次（组）的测试，每次都更换不同的验证集。即得到5组模型的结果，取平均值作为最终结果。又称**5折交叉验证**。(通常情况加下采取十折验证)
![imagedaa05.png](https://miao.su/images/2019/07/05/imagedaa05.png)

问题：那么这个只是对于参数得出更好的结果，那么怎么选择或者调优参数呢？

### 超参数搜索-网格搜索(Grid Search)

通常情况下，有很多参数是需要**手动指定**的(**如k-近邻算法中的k值**)，这种叫做**超参数**。但是手动过程繁杂，所以需要对模型预设几种超参数组合。**每组超参数都采用交叉验证来进行评估。最后选出最优参数组合建立模型。**
![image7eab3.png](https://miao.su/images/2019/07/05/image7eab3.png)

### 模型选择与调优

* `sklearn.model_selection.GridSearchCV(estimator, param_grid=None, cv=None)`

  * 对估计器的指定参数值进行详尽搜索
  * `estimator`：估计器对象
  * `param_grid`：估计器参数`(dict){"n_neighbors":[1, 3, 5]}`
  * `cv`：指定几折交叉验证

  返回结果：

  * `fit`：输入训练数据
  * `score`：准确率
  * 结果分析：
    * `best_score_`：在交叉验证中验证的最好结果
    * `beste_stmator_`：最好的参数模型
    * `cv_results_`：每次交叉验证后的验证集准确率结果和训练集准确率结果



## 朴素贝叶斯算法

### 什么是朴素贝叶斯算法？

![imagebf36b.png](https://miao.su/images/2019/07/06/imagebf36b.png)
![image42c54.png](https://miao.su/images/2019/07/06/image42c54.png)

### 概率基础

#### 概率定义

$P(x)[0,1]$

#### 引例：

![imagec6f6a.png](https://miao.su/images/2019/07/06/imagec6f6a.png)
![imagebf04f.png](https://miao.su/images/2019/07/06/imagebf04f.png)

#### 条件概率和联合概率

- 联合概率：包含多个条件，且所有条件同时成立的概率
  - 记作：P(A,B)
  - 特性：P(A, B) = P(A)P(B)
- 条件概率：就是事件A在另外一个事件B已经发生条件下的发生概率
  - 记作：P(A|B)
  - 特性：P(A1, A2|B) = P(A1|B)P(A2|B)

> 注意：此条件概率的成立，**是由于A1,A2相互独立的结果**(记忆)

这样我们计算结果为：

```python
p(程序员, 匀称) =  P(程序员)P(匀称) =3/7*(4/7) = 12/49 
P(产品, 超重|喜欢) = P(产品|喜欢)P(超重|喜欢)=1/2 *  1/4 = 1/8
```

**那么，我们知道了这些知识之后，继续回到我们的主题中。朴素贝叶斯如何分类，这个算法经常会用在文本分类，那就来看文章分类是一个什么样的问题？**
$$
P(科技|文章)，P(娱乐|文章)
$$

### 贝叶斯公式

$$
P(C|W)=\frac{P(W|C)P(C)}{P(W)}
$$

*注：W为给定文档的特征值（频数统计，预测文档提供）， C为文档类别*

**那么这个公式如果应用在文章分类的场景中，我们可以这样看：**
公式可以理解为：
$$
P(C|F1,F2,...)=\frac{P(F1,F2,...|C)P(C)}{P(F1,F2,...)}
$$
其中C可以是不同的类别。公式分为三个部分：

- P(C)：每个文档类别的概率(某文档类别数／总文档数量)
- P(W│C)：给定类别下特征（被预测文档中出现的词）的概率
  - 计算方法：P(F1│C)=Ni/N （训练文档中去计算）
    - Ni为该F1词在C类别所有文档中出现的次数
    - N为所属类别C下的文档所有词出现的次数和
- P(F1,F2,…) 预测文档中每个词的概率

#### 文章分类计算

* 假设我们从**训练数据集**的到如下信息

![imagedde2b.png](https://miao.su/images/2019/07/06/imagedde2b.png)

* 计算结果

  ```python
  科技：P(科技|影院,支付宝,云计算) = 𝑃(影院,支付宝,云计算|科技)∗P(科技)=(8/100)∗(20/100)∗(63/100)∗(30/90) = 0.00456109
  
  娱乐：P(娱乐|影院,支付宝,云计算) = 𝑃(影院,支付宝,云计算|娱乐)∗P(娱乐)=(56/121)∗(15/121)∗(0/121)∗(60/90) = 0
  ```

*思考：我们计算出某个结果为零，合适么？*

#### 拉普拉斯平滑系数

目的：防止计算出的分类概率为0
$$
P(F_1|C)=\frac{N_i+\alpha}{N+\alpha m}
$$
$\alpha$为指定的系数，一般为1，$m$为训练文档中统计出的**特征词**的个数

```python
P(娱乐|影院,支付宝,云计算) =P(影院,支付宝,云计算|娱乐)P(娱乐) =P(影院|娱乐)*P(支付宝|娱乐)*P(云计算|娱乐)P(娱乐)=(56+1/121+4)(15+1/121+4)(0+1/121+1*4)(60/90) = 0.00002
```

#### API

* `sklearn.naive_bayes.MultinomiaNB(alpha = 1.0)`
  * 朴素贝叶斯分类
  * `alpha`：拉普拉斯平滑系数

### 案例：20类新闻分类

![image7845a.png](https://miao.su/images/2019/07/06/image7845a.png)

#### 分析

* 分隔数据集
* tfidf进行的特征抽取
* 朴素贝叶斯预测

#### 代码

```python
  1 #!/usr/bin/env python
  2 # coding=utf-8
  3 from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
  4 from sklearn.feature_extraction import DictVectorizer
  5 from sklearn.preprocessing import StandardScaler
  6 from sklearn.datasets import fetch_20newsgroups
  7 from sklearn.naive_bayes import MultinomialNB
  8 from sklearn.model_selection import train_test_split, GridSearchCV
  9 import pandas as pd
 10 
 11 
 12 def nbcls():
 13     """朴素贝叶斯对20类新闻进行分类"""
 14     news = fetch_20newsgroups(subset="all")
 15     # print(news)
 16 
 17     # 1.分割数据集
 18     x_train, x_test, y_train, y_test = train_test_split(news.data, news.target, test_size=0.3)
 19 
 20     # 2.进行文本特征抽取，让算法能够知道特征值
 21     # 实例化tfidf
 22     tfidf = TfidfVectorizer()
 23     # 对训练集特征抽取
 24     x_train = tfidf.fit_transform(x_train)
 25     # 对测试集特征抽取
 26     # 注意，此处不再是fit_transform，因我们是根据训练集统计出的特征来预测测试集
 27     x_test = tfidf.transform(x_test)
 28 
 29     # 进行朴素贝叶斯算法预测
 30     mlb = MultinomialNB(alpha=1.0)
 31     mlb.fit(x_train, y_train)
 32     # 预测，准确率
 33     print("预测测试集当中的文档类别是:", mlb.predict(x_test)[:50])
 34     print("测试集当中的文档真实类别是:", y_test[:50])
 35 
 36     # 得出准确率
 37     print("文档分类的准确率为:", mlb.score(x_test, y_test))
 38 
 39 
 40     return None 
 41 
 42 
 43 
 44 if __name__ == '__main__':
 45     nbcls()
```

#### 总结

* 优点：
  * 朴素贝叶斯模型发源于古典数学理论，有稳定的分类效率（古典概型）
  * 对缺失数据不太敏感，算法也比较简单，常用于文本分类
  * 分类准确度高、速度快
* 缺点：
  * 由于使用了样本属性独立性的假设，所以如果特征属性有关联时其效果不好（即事件相互独立）



## 决策树

### 认识决策树

决策树思想的来源非常朴素，程序设计中的条件分支结构就是`if-then`结构，最早的决策树就是利用这类结构分隔数据的一种分类学习方法 。

可以通过下面的例子来理解：
![image03c6b.png](https://miao.su/images/2019/07/06/image03c6b.png)

### 决策树分类原理详解

为了更好的理解决策树具体怎么分类的，我们通过一个问题例子：
![imageb30c7.png](https://miao.su/images/2019/07/06/imageb30c7.png)

可能你的划分是这样的：
![image981a5.png](https://miao.su/images/2019/07/06/image981a5.png)

真正的决策树是这样划分的：
![image751de.png](https://miao.su/images/2019/07/06/image751de.png)

### 原理

* 信息熵、信息增益等
* 需要用到信息论的知识

**决策树的一般流程**

1. 收集数据：可以使用任何方法
2. 准备数据：树构造算法只适用于标称型数据，因此数值型数据必须离散化
3. 分析数据：可以使用任何方法，构造树完成之后，我们应该检查图形是否符合预期
4. 训练算法：构造树的数据结构
5. 测试算法：使用经验树计算错误率
6. 使用算法：此步骤可以适用于任何监督学习算法，而使用决策树可以更好地理解数据的内在含义



### 信息熵

那来玩个猜测游戏，猜猜这32支球队那个是冠军。并且猜测错误付出代价。每猜错一次给一块钱，告诉我是否猜对了，那么我需要掏多少钱才能知道谁是冠军？ （前提是：不知道任意球队的信息、历史比赛记录、实力等）

![image7cfda.png](https://miao.su/images/2019/07/06/image7cfda.png)

**为了使代价最小，可以使用二分法猜测：**

我可以把球编上号，从1到32，然后提问：冠 军在1-16号吗？依次询问，只需要五次，就可以知道结果。
![image45da4.png](https://miao.su/images/2019/07/06/image45da4.png)

我们来看这个式子：

* 32支球队，$log_232=5$
* 64支球队，$log_264=6$

![image342d9.png](https://miao.su/images/2019/07/06/image342d9.png)

**香农指出，它的准确信息量应该是，p为每个球队获胜的概率（假设概率相等，都为1/32）， 我们不用钱去衡量这个代价了，香农指出用比特：**
$$
H=-(p_1log_2p_1+p_2log_2p_2+p_3log_2p_3+...+p_{32}log_2p_{32})=-log_232
$$

#### 信息熵的定义

* $H$ 的专业术语称之为信息熵，单位为比特。

$$
H=-\displaystyle\sum_{i=1}^np(x_i)log_2p(x_i)
$$

* “谁是世界杯冠军”的概率相同，则结果为5bit
* 只要有一个元素不同，那么这个结果就一定小于5bit
* 即，只要概率发生变化，信息熵都比5bit小

#### 总结（重要）

* **信息熵和消除不确定性是相联系的**
  当我们得到额外信息（球队历史比赛情况等等）越多的话，那么我们猜测的代价越小（猜测的不确定性减小）
* *问题：回到我们前面的贷款案例，怎么去划分？可以利用当得知某个特征（比如是否有房子）之后，我们能够减少的不确定性大小，越大我们可以认为这个特征很重要。那怎么去衡量减少的不确定性大小呢？*

### 决策树的划分依据之一——信息增益

#### 定义与公式

特征A对训练数据集D的信息增益`g(D, A)`，定义为集合D的信息熵`H(D)`与特征A给定条件下D的信息条件熵`H(D|A)`之差，即公式为：
$$
g(D,A)=H(D)-H(D|A)
$$
公式的详细解释：
![image222c1.png](https://miao.su/images/2019/07/08/image222c1.png)

注：信息增益表示得知特征X的信息而使得类Y的信息熵减少的程度

#### 贷款特征重要计算

![imageb30c7.png](https://miao.su/images/2019/07/06/imageb30c7.png)

* 我们以年龄为特征来计算：

  ```markdown
  1、g(D, 年龄) = H(D) -H(D|年龄) = 0.971-[5/15H(青年)+5/15H(中年)+5/15H(老年]
  
  2、H(D) = -(6/15log(6/15)+9/15log(9/15))=0.971
  
  3、H(青年) = -(3/5log(3/5) +2/5log(2/5))
  H(中年)=-(3/5log(3/5) +2/5log(2/5))
  H(老年)=-(4/5og(4/5)+1/5log(1/5))
  ```

  我们以A1、A2、A3、A4代表年龄、有工作、有自己的房子和贷款情况。最终计算的结果g(D, A1) = 0.313, g(D, A2) = 0.324, g(D, A3) = 0.420,g(D, A4) = 0.363。所以我们选择A3 作为划分的第一个特征。这样我们就可以一棵树慢慢建立

### 决策树的三种算法实现

当然决策树的原理不止信息增益这一种，还有其他方法。但是原理都类似，我们就不去举例计算。

* ID3
  * 信息增益最大准则
* C4.5
  * 信息增益比 最大的准则
* CART
  * 分类树：基尼系数 最小的准则 在sklearn中可以选择划分的默认原则
  * 优势：划分更加细致（从后面例子的树显示来理解）

### 决策树API

- `class sklearn.tree.DecisionTreeClassifier(criterion=’gini’, max_depth=None,random_state=None)`
  - 决策树分类器
  - `criterion`:默认是`gini`系数，也可以选择信息增益的熵`entropy`
  - `max_depth`:树的深度大小
  - `random_state`:随机数种子
- 其中会有些**超参数：max_depth:树的深度大小**
  - 其它超参数我们会结合随机森林讲解

### 案例：泰坦尼克号乘客生存预测

- 泰坦尼克号数据

在泰坦尼克号和titanic数据帧描述泰坦尼克号上的个别乘客的生存状态。这里使用的数据集是由各种研究人员开始的。其中包括许多研究人员创建的旅客名单，由Michael A. Findlay编辑。我们提取的数据集中的特征是票的类别，存活，乘坐班，年龄，登陆，home.dest，房间，票，船和性别。

1. **乘坐班是指乘客班（1，2，3），是社会经济阶层的代表**
2. **其中age数据存在缺失**

*数据：http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt*

#### 分析

- 选择我们认为重要的几个特征 ['pclass', 'age', 'sex']
- 填充缺失值
- 特征中出现类别符号，需要进行one-hot编码处理(DictVectorizer)
  - x.to_dict(orient="records") 需要将数组特征转换成字典数据
- 数据集划分
- 决策树分类预测

#### 代码

```python
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
```

*由于决策树类似一个树的结构，我们可以保存到本地显示*

### 保存树的结构到dot文件

- 1、`sklearn.tree.export_graphviz()`该函数能够导出DOT格式
  - tree.export_graphviz(estimator,out_file='tree.dot’,feature_names=[‘’,’’])
- 2、工具:(能够将dot文件转换为pdf、png)
  - 安装graphviz
  - ubuntu: sudo apt-get install graphviz Mac:brew install graphviz
- 3、运行命令
  - 然后我们运行这个命令
  - dot -Tpng tree.dot -o tree.png

```python
export_graphviz(dc, out_file="./tree.dot", feature_names=['age', 'pclass=1st', 'pclass=2nd', 'pclass=3rd', '女性', '男性'])
```

### 决策树总结

- 优点：
  - 简单的理解和解释，树木可视化。
- 缺点：
  - 决策树学习者可以创建不能很好地推广数据的过于复杂的树，这被称为**过拟合**。
- 改进：
  - 减枝cart算法(决策树API当中已经实现，随机森林参数调优有相关介绍)
  - **随机森林**

*注：企业重要决策，由于决策树很好的分析能力，在决策过程应用较多， 可以选择特征*

## 集成学习方法之随机森林

### 什么是集成学习方法

集成学习通过建立几个模型组合的来解决单一预测问题。它的工作原理是**生成多个分类器/模型**，各自独立地学习和作出预测。*这些预测最后结合成组合预测，因此优于任何一个单分类的做出预测。*

### 什么是随机森林

在机器学习中，**随机森林是一个包含多个决策树的分类器**，并且其输出的类别是由个别树输出的类别的众数而定。
*例如, 如果你训练了5个树, 其中有4个树的结果是True, 1个数的结果是False, 那么最终投票结果就是True*

### 随机森林生成的过程

根据下列算法而建造每棵树：

- 用N来表示训练用例（样本）的个数，M表示特征数目。
  - 1、一次随机选出一个样本，重复N次， （有可能出现重复的样本）
  - 2、随机去选出m个特征, m <<M，建立决策树
- 采取bootstrap抽样

### 为什么采用BootStrap抽样

- 为什么要随机抽样训练集？　　
  - 如果不进行随机抽样，每棵树的训练集都一样，那么最终训练出的树分类结果也是完全一样的
- 为什么要有放回地抽样？
  - 如果不是有放回的抽样，那么每棵树的训练样本都是不同的，都是没有交集的，这样每棵树都是“有偏的”，都是绝对“片面的”（当然这样说可能不对），也就是说每棵树训练出来都是有很大的差异的；而随机森林最后分类取决于多棵树（弱分类器）的投票表决。

### API

- class sklearn.ensemble.RandomForestClassifier(n_estimators=10, criterion=’gini’, max_depth=None, bootstrap=True, random_state=None, min_samples_split=2)
  - 随机森林分类器
  - n_estimators： integer，optional（default = 10）森林里的树木数量120,200,300,500,800,1200
  - criteria： string，可选（default =“gini”）分割特征的测量方法
  - max_depth： integer或None，可选（默认=无）树的最大深度 5,8,15,25,30
  - max_features="auto”,每个决策树的最大特征数量
    - If "auto", then `max_features=sqrt(n_features)`.
    - If "sqrt", then `max_features=sqrt(n_features)` (same as "auto").
    - If "log2", then `max_features=log2(n_features)`.
    - If None, then `max_features=n_features`.
  - bootstrap： boolean，optional（default = True）是否在构建树时使用放回抽样
  - min_samples_split:节点划分最少样本数
  - min_samples_leaf:叶子节点的最小样本数

- 超参数：n_estimator, max_depth, min_samples_split,min_samples_leaf

### 泰坦尼克再现代码

```python
#!/usr/bin/env python
# coding=utf-8
from sklearn.feature_extraction import DictVectorizer
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier


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
    # dec = DecisionTreeClassifier(max_depth=5)

    # dec.fit(x_train, y_train)

    # print("预测的准确率为：", dec.score(x_test, y_test))

    # 导出到dot文件
    # export_graphviz(dec, out_file="./test.dot",
    # feature_names=['age', 'pclass=1st', 'pclass=2nd', 'pclass=3rd', '女性', '男性'])

    # 5.随机森林进行预测
    # 实例化
    rdtree = RandomForestClassifier()

    # 构造超参数的字典
    param = {"n_estimators":[120, 200],
             "max_depth": [5, 8],
             "min_samples_split": [2, 3]}
    gc = GridSearchCV(rdtree, param_grid=param, cv=2)
    gc.fit(x_train, y_train)
    print("随机森林的准确率:", gc.score(x_test, y_test))
    print("交叉验证选择的参数:", gc.best_estimator_)
    return None


if __name__ == '__main__':
    decision()
```

### 总结

- 在当前所有算法中，具有极好的准确率
- 能够有效地运行在大数据集上，处理具有高维特征的输入样本，而且不需要降维
- 能够评估各个特征在分类问题上的重要性