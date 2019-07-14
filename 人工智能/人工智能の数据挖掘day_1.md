# 人工智能の数据挖掘

[TOC]

## 数据挖掘基础环境安装与使用

* 完成数据挖掘基础阶段的所有环境安装
* 应用 `jupyter notebook` 完成代码编写运行

### 库的安装

整个数据挖掘基础阶段会用到`Matplotlib、Numpy、Pandas、Ta-Lib`等库，为了统一版本号在环境中使用，将所有的库及其版本放到了文件`requirements.txt`中，然后统一安装

*新建一个用于人工智能环境的虚拟环境*

```shell
mkvirtualenv -p python3 ai
```

```shell
matplotlib==2.0.2
numpy==1.14.2
pandas==0.20.3
TA-Lib==0.4.16
tables==3.4.2
jupyter==1.0.0
```

使用pip命令安装：

```shell
pip install -r requirements.txt
```

## Jupyter notebook使用

### 介绍

Jupyter项目是一个非盈利的开源项目，源于2014年的ipython项目，因为它主键发展为支持跨所有编程语言的交互式数据科学和科学计算

* Jupyter Notebook，原名Ipython Notebook，是Ipython的加强网页版，一个开源Web应用程序
* 名字源自Julia、Python和R（数据科学的三种开源语言）
* **.ipynb**文件格式是用于计算型叙述的**JSON文档格式**的正式规范
  ![image51c1e.png](https://miao.su/images/2019/06/23/image51c1e.png)

### 为什么要使用Jupyter

* 传统软件开发：工程/目标明确
  * 需求分析、设计架构、开发模块、测试
* 数据挖掘：艺术/目标不明确
  * 目的是具体的洞察目标，而不是机械的完成任务
  * 通过执行代码来理解问题
  * 迭代式地改进代码来改进解决方法

实时运行的代码、叙事性的文本和可视化被整合在一起，方便使用代码和数据来讲述故事。

### Jupyter使用

界面启动、新建

**本机打开**

```shell
# 进入虚拟环境
workon ai
# 输入命令
jupyter notebook
```

* 本地notebook的默认URL为：`http://localhost:8888`
* 需要配置 `~/.jupyter/jupyter notebook config`
* 想让notebook打开指定目录，只要进入此目录后执行命令即可

![image0ab39.png](https://miao.su/images/2019/06/24/image0ab39.png)

新建notebook文档：
![image2f233.png](https://miao.su/images/2019/06/24/image2f233.png)

内容界面：
![image75cf8.png](https://miao.su/images/2019/06/24/image75cf8.png)

- 标题栏：点击标题（如Untitled）修改文档名
- 菜单栏
  - 导航-File-Download as，另存为其他格式
  - 导航-Kernel
    - Interrupt，中断代码执行（程序卡死时）
    - Restart，重启Python内核（执行太慢时重置全部资源）
    - Restart & Clear Output，重启并清除所有输出
    - Restart & Run All，重启并重新运行所有代码

### cell

#### cell操作

cell：一堆In Out会话被视作一个单元代码，称为cell
Jupyter支持两种模式：

* 编辑模式（Enter）
  * 命令模式下`回车Enter`或`鼠标双击`cell进入编辑模式
  * 可以**操作cell内文本**或代码，剪切/复制/粘贴移动等操作
* 命令模式（Esc）
  * 按`Esc`退出编辑，进入命令模式
  * 可以**操作cell单元本身**进行剪切/复制/粘贴/移动等操作

![imagefe989.png](https://miao.su/images/2019/06/25/imagefe989.png)

#### 快捷键操作cell

* 两种模式通用跨快捷键
  * `Shift+Enter`，执行本单元代码，并跳转下一单元
  * `Ctrl+Enter`，执行本单元代码，留在本单元
    cell行号签的`*`，表示代码正在运行
* 命令模式：按`Esc`进入
  * `Y`，cell切换到Code模式
  * `M`，cell切换到Markdown模式
  * `A`，在当前cell的上面添加cell
  * `B`，在当前cell的下面添加cell
  * `双击D`，删除当前cell
  * `Z`，回退
  * `L`，为当前cell加上行号
  * `Ctrl+Shift+P`，对话框输入命令直接运行
  * 快速跳转到首个cell，`Ctrl+Home`
  * 快速跳转到最后一个cell,`Ctrl+End`
* 编辑模式：按`Enter`进入
  * 多光标操作：`Ctrl键点击鼠标`
  * 回退：`Ctrl+Z`
  * 重做：`Ctrl+Y`
  * 补全代码：变量、方法后跟`Tab键`
  * 为一行或多行代码添加注释：`Ctrl+/`
  * 屏蔽自动输出信息：可在最后一条语句之后加一个分号

## Matplotlib

### 绘图架构

#### 什么是Matplotlib

![imageb3aad.png](https://miao.su/images/2019/06/25/imageb3aad.png)

* 是专门用于开发2D图表（包括3D图表）
* 使用起来及其简单
* 以渐进、交互式方式实现数据可视化

#### 为什么要学习Matplotlib

可视化是整个数据挖掘中的关键辅助工具，可以清晰的理解数据，从而调整我们的分析方法。

* 能将数据进行可视化，更直观的呈现
* 使数据更加客观、更具说服力

例如下面两个图为数字展示和图形展示：
![imaged9e44.png](https://miao.su/images/2019/06/25/imaged9e44.png)

我们先来简单画一个图看一下效果：
![image4bca8.png](https://miao.su/images/2019/06/25/image4bca8.png)

### Matplotlib框架结构

matplotlib框架分为三层，这三层构成了一个栈，上层可以调用下层。
![image40eea.png](https://miao.su/images/2019/06/25/image40eea.png)

#### 后端层

matplotlib的底层，实现了大量的抽象接口类，这些API用来在底层实现图形元素的一个个类。

* FigureCanvas对象 实现了绘图区域这一概念
* Renderer对象 在FigureCanvas上绘图

#### 美工层

图形中所有能看到的元素都属于Artist对象，即标题、轴标签、刻度等组成图形的所有元素都是Artist对象的实例：

* Figure：指整个图形（包括所有的元素，比如标题、线等）
* Axes（**坐标系**）：数据的绘图区域
* Axis（**坐标轴**）：坐标系中的一条轴，包含大小限制、刻度和刻度标签

特点为：

* 一个figure（图）可以包含多个axes（坐标系），但是一个axes只能属于一个figure
* 一个axes（坐标系）可以包含多个axis（坐标轴），包含两个即为2d坐标系

![imagedc781.png](https://miao.su/images/2019/06/25/imagedc781.png)

#### 脚本层

主要用于可视化编程，pyplot模块可以提供给我们一个与matplotlib打交道的接口。可以只通过调用pyplot模块的函数从而操作整个程序包，来绘制图形。

* 操作或者改动Figure对象，例如创建Figure对象
* 大部分工作是处理样本文件的图形与坐标的生成

## 折线图与基础绘图功能

Parts of  a Figure
![imagedfe6d.png](https://miao.su/images/2019/06/25/imagedfe6d.png)

### 折线图绘制与保存图片

为了更好的去理解所有基础绘图功能，我们通过天气温度变化的绘图来融合所有的基础API使用。

#### matplotlib.pyplot模块

matplotlib.pyplot包含了一系列类似于matlab的画图函数。它的函数*作用于当前图形（figure）的当前坐标系（axes）*。

```python
# 为方便简介,简称为plt
import matplotlib.pyplot as plt
```

#### 折线图绘制与显示

*展现上海一周的天气，比如从星期一到星期日的天气温度如下*：
![image4a32d.png](https://miao.su/images/2019/06/25/image4a32d.png)

可以看到这样去显示效果并不好，图形的大小等等，所以我们可以通过加入更多的功能。

#### 修改图形大小与图片保存

```python
plt.figure(figsize=(x, x), dpi = x)
	figsize：指定图的长和宽
	dpi：指定图像的清晰度
	返回fig对象
plt.savefig(path)
```

![image31b2c.png](https://miao.su/images/2019/06/25/image31b2c.png)

### 温度变化表示

需求：画出某城市11点到12点1小时内每分钟的温度变化折线图，温度范围在15度到18度

效果如下：

![image853f9.png](https://miao.su/images/2019/06/25/image853f9.png)

#### 构造数据、显示

![imagedc4fd.png](https://miao.su/images/2019/06/26/imagedc4fd.png)

#### 自定义 $x$轴和 $y$轴刻度以及中文显示

* plt.xticks(x, **kwargs)
  x：要显示的刻度值

* plt.yticks(y, **kwargs)
  y：要显示的刻度值

增加以下两行代码构造中文列表的字符串

```python
  x_ch = ["11点{}分".format(i) for i in x]
  y_ticks = range(40)
```



修改x,y坐标的刻度:

```python
  plt.xticks(x[::5], x_ch[::5])
  plt.yticks(y_ticks[::5])
```

#### 增加坐标轴信息

```python
plt.xlabel("时间")
plt.ylabel("温度")
plt.title("中午11点0分到12点之间的温度变化图示")
```

![image21551.png](https://miao.su/images/2019/07/12/image21551.png)

#### 再添加一个城市的温度变化

收集到北京当天温度变化情况，温度在1度到3度。怎么去添加另一个在同一坐标系当中的不同图形，*其实很简单只需要再次plot即可*，但是需要区分线条，如下显示

![image8bfad.png](https://miao.su/images/2019/07/12/image8bfad.png)

```python
 再添加一个城市
# 生成北京的温度
y_beijing = [random.uniform(1, 3) for i in x]

# 画折线图
plt.plot(x, y_shanghai, label="上海")
# 使用plot可以多次画多个折线
plt.plot(x, y_beijing, color='r', linestyle='--', label="北京")

# 添加图形注释
plt.legend(loc="best")


# 画折线图
plt.plot(x, y_shanghai, label='上海')
plt.show()
```

#### 自定义图形风格

| 颜色字符 |    风格字符    |
| :------: | :------------: |
|  r 红色  |     - 实线     |
|  g 绿色  |    - - 虚线    |
|  b 蓝色  |   -. 点划线    |
|  w 白色  |    : 点虚线    |
|  c 青色  | ' ' 留空、空格 |
|  m 洋红  |                |
|  y 黄色  |                |
|  k 黑色  |                |

#### 添加图例注释

```visual basic
plt.legend(loc="best")
```

![imagee006a.png](https://miao.su/images/2019/07/12/imagee006a.png)

### 多坐标系显示-subplots

如果我们想要将上海和北京的天气图显示在同一个图的不同坐标系当中，效果如下：

可以通过subplots函数实现(旧的版本中有subplot，使用起来不方便)，推荐subplots函数。
`matplotlib.pyplot.subplots(nrows=1, ncols=1, **fig_kw)` 创建一个带有多个坐标系的图：

```python
%matplotlib inline
import random
import matplotlib.pyplot as plt

from matplotlib.pylab import mpl
mpl.rcParams['font.sans-serif'] = ['arial unicode ms']   #显示中文
mpl.rcParams['axes.unicode_minus']=False       #显示负号


# 两个城市的温度，在多个坐标系中显示
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20,8))
"""fig是画布，ax是坐标系，通过ax[0],ax[1]获取"""

# 准备x，y轴数据
x = range(60)
y_shanghai = [random.uniform(15, 18) for i in x]
y_beijing = [random.uniform(8, 10) for i in x]

# 增加下面两行到代码，编辑要显示的中文
x_ch = ["11点{}分".format(i) for i in x]
y = range(40)

"""当在多个ax中画图的时候，刻度、标签、必须在相应的坐标系里面制定"""
# 上海显示
ax[0].plot(x, y_shanghai, label="上海")
# 显示上海的温度
ax[1].plot(x, y_beijing, color='r', linestyle='--', label="北京")


# 修改x,y坐标的刻度，指定对应的中文（注意，两者的步长必须一样）
# plt是对于整体处理，而ax是对于每一个坐标系指定
# plt.xticks(x[::5], x_ch[::5])
# plt.yticks(y[::5])
ax[0].set_xticks(x[::5], x_ch[::5])
ax[1].set_xticks(x[::5], x_ch[::5])
ax[0].set_yticks(y[::5])
ax[1].set_yticks(y[::5])

ax[0].set_xlabel("时间")
ax[1].set_xlabel("时间")
ax[0].set_ylabel("温度")
ax[1].set_ylabel("温度")

ax[0].set_title("中午11点0分到12点之间的温度变化图示")
ax[1].set_title("中午11点0分到12点之间的温度变化图示")

ax[0].legend(loc="best")
ax[1].legend(loc="best")


# 画折线图
plt.show()
plt.savefig('test2.png')
```

```python
Parameters:    
nrows, ncols : int, optional, default: 1, Number of rows/columns of the subplot grid.
**fig_kw : All additional keyword arguments are passed to the figure() call.
Returns:    
fig : 图对象
ax : 
    设置标题等方法不同：
    set_xticks
    set_yticks
    set_xlabel
    set_ylabel
```

关于axes子坐标系的更多方法：参考https://matplotlib.org/api/axes_api.html#matplotlib.axes.Axes

### 折线图应用场景

- 呈现公司产品(不同区域)每天活跃用户数
- 呈现app每天下载数量
- 呈现产品新功能上线后,用户点击次数随时间的变化

### 总结

<u>开头的这几个目标应用全都很重要</u>

- 知道如何解决中文显示问题
- 知道matplotlib的图结构
- 应用figure实现创建绘图区域大小
- 应用plot实现折线图的绘制
- 应用title,xlabel,ylabel实现标题以及x,y轴名设置
- 应用xticks,yticks实现axes的刻度设置和标注
- 应用savefig实现图形的本地保存
- 应用grid实现显示网格应用axis实现图像形状修改
- 应用legend实现图形标注信息显示
- 应用plt.subplots实现多坐标系的创建
- 知道如何设置多个axes的标题、刻度

