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
  * y 数据集的标签
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

