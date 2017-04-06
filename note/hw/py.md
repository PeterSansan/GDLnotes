安装命令最好加上sudo

## 一、pip切换镜像源

一定要换，不换不知道，一换吓一跳，不掉的话可能安装模块时下载速度很慢:v:

新版ubuntu要求使用https源，要注意。
```
清华：https://pypi.tuna.tsinghua.edu.cn/simple
阿里云：http://mirrors.aliyun.com/pypi/simple/
中国科技大学 https://pypi.mirrors.ustc.edu.cn/simple/
华中理工大学：http://pypi.hustunique.com/
山东理工大学：http://pypi.sdutlinux.org/ 
豆瓣：http://pypi.douban.com/simple/
```
#### 1）临时使用

可以在使用pip的时候加参数-i https://pypi.tuna.tsinghua.edu.cn/simple

例如：`pip install -i https://pypi.tuna.tsinghua.edu.cn/simple pyspider`，这样就会从清华这边的镜像去安装pyspider库。
 
#### 2）永久修改，一劳永逸

Linux下，修改 ~/.pip/pip.conf (没有就创建一个文件夹及文件。文件夹要加“.”，表示是隐藏文件夹)

内容如下：
```
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
[install]
trusted-host=mirrors.aliyun.com
```
windows下，直接在user目录中创建一个pip目录，如：C:\Users\xx\pip，新建文件pip.ini。内容同上。

---
## 二、安装具体版本软件

```
$ pip install SomePackage            # latest version
$ pip install SomePackage==1.0.4     # specific version
$ pip install 'SomePackage>=1.0.4'     # minimum version
```

## 三、conda的使用
conda的使用我只记住了几个命令，如
```
conda info
conda list
conda info --envs or conda info -e  # 列出环境变量
```
不过呢，安装包相关的命令可以用pip用代替，conda简明版的命令可以看`conda-cheatsheet.pdf`，复杂的看官网吧

## 四、python常用函数
```
del variable    # 删除变量
type(variable)  # 查看变量的数据类型
time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())) # 输出当前时间，要import 

for t in reversed(xrange(100):
  print t  # 输出序号从大到小
  
for t in [a,b]
  print t  # 取的是是

print("{}".format(变量))
```
### 一个很好用的语法
```
[w for w in text if confition]  # 判断返加一个子元组

```
### for与zip
```
x = [1, 2, 3]

y = [4, 5, 6]

z = [7, 8, 9]

xyz = zip(x, y, z)

print xyz

```
运行结果：
```
[(1, 4, 7), (2, 5, 8), (3, 6, 9)]
```
```
  for param, dparam, mem in zip([Wxh, Whh, Why, bh, by],    # 这个结果真是强大
                                [dWxh, dWhh, dWhy, dbh, dby], 
                                [mWxh, mWhh, mWhy, mbh, mby]):
    mem += dparam * dparam   # +=能对Wxh等参数赋值，=不行
    param += -learning_rate * dparam / np.sqrt(mem + 1e-8) # adagrad update
```
### xrange与range的区别
range用法与range完全相同，所不同的是生成的不是一个list对象，而是一个生成器。
```
>>> xrange(5)
xrange(5)
>>> list(xrange(5))
[0, 1, 2, 3, 4]
>>> xrange(1,5)
xrange(1, 5)
>>> list(xrange(1,5))
[1, 2, 3, 4]
>>> xrange(0,6,2)
xrange(0, 6, 2)
>>> list(xrange(0,6,2))
[0, 2, 4]
```
> 由上面的示例可以知道：要生成很大的数字序列的时候，用xrange会比range性能优很多，因为不需要一上来就开辟一块很大的内存空间
## 五、numpy的使用

### numpy简单的生成随机数函数
```
rand(d0, d1, ..., dn)	Random values in a given shape.
randn(d0, d1, ..., dn)	Return a sample (or samples) from the “standard normal” distribution.
randint(low[, high, size, dtype])	Return random integers from low (inclusive) to high (exclusive).
random_integers(low[, high, size])	Random integers of type np.int between low and high, inclusive.
random_sample([size])	Return random floats in the half-open interval [0.0, 1.0).
random([size])	Return random floats in the half-open interval [0.0, 1.0).
ranf([size])	Return random floats in the half-open interval [0.0, 1.0).
sample([size])	Return random floats in the half-open interval [0.0, 1.0).
choice(a[, size, replace, p])	Generates a random sample from a given 1-D array
bytes(length)	Return random bytes.
```
###  python numpy矩阵信息，shape，size，dtype
```
import numpy as np  
from numpy import random  
matrix1 = random.random(size=(2,4))  
#矩阵每维的大小  
print matrix1.shape  
```
```
#矩阵所有数据的个数  
print matrix1.size  
```
```
#矩阵每个数据的类型  
print matrix1.dtype  
```
### numpy矩阵乘法
```
a = np.ones([5,5])
b = np.ones([5,5])
np.dot(a,b)
```
### 其他
```
numpy.clip(a, a_min, a_max, out=None)# 截断数据，out是输入的变量

numpy.arange(-1,1,0.01) # 每隔0.01隔出一个离散点，像matlab
```
**np.vstack**
```
>>> a = np.array([1, 2, 3])
>>> b = np.array([2, 3, 4])
>>> np.vstack((a,b))
array([[1, 2, 3],
       [2, 3, 4]])
>>>
>>> a = np.array([[1], [2], [3]])
>>> b = np.array([[2], [3], [4]])
>>> np.vstack((a,b))
array([[1],
       [2],
       [3],
       [2],
       [3],
       [4]])
```
## 六、Argparse简易使用
### 1.简介

argparse是python用于解析命令行参数和选项的标准模块，用于代替已经过时的`optparse`模块。`argparse`模块的作用是用于解析命令行参数，例如`python parseTest.py input.txt output.txt --user=name --port=8080`。
### 2.使用步骤
- 1：import argparse
- 2：parser = argparse.ArgumentParser()
- 3：parser.add_argument()
- 4：parser.parse_args()
 
> 解释：首先导入该模块；然后创建一个解析对象；然后向该对象中添加你要关注的命令行参数和选项，每一个`add_argument`方法对应一个你要关注的参数或选项；最后调用`parse_args()`方法进行解析；解析成功之后即可使用.另外，`--help`或`-h`是唯一预设的参数，如可以用`python a.py -h`查看

### 3.固定参数
```
# a.py 

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("square", help="display a square of a given number",
                    type=int)
args = parser.parse_args()
print args.square**2
```
运行结果：

![img](http://ogtxggxo6.bkt.clouddn.com/k.png)

### 4.可选参数
```
# a.py

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-v", "--verbose", help="increase output verbosity",
                    action="store_true")
args = parser.parse_args()
if args.verbose:
    print "verbosity turned on"
```
> 现在多加一个标志来替代必须给出一些值，并修改了名称来表达我们的意思。注意现在指定了一个新的关键词action，并且赋值为store_ture。如果指定了这个可选参数，args.verbose就赋值为True,否则就为False。

![img](http://ogtxggxo6.bkt.clouddn.com/l.png)

## 七、[不可不知的python模块：collections](http://www.zlovezl.cn/articles/collections-in-python/)

Python作为一个“内置电池”的编程语言，标准库里面拥有非常多好用的模块。比如今天想给大家 介绍的 collections 就是一个非常好的例子。

基本介绍
我们都知道，Python拥有一些内置的数据类型，比如str, int, list, tuple, dict等， collections模块在这些内置数据类型的基础上，提供了几个额外的数据类型：

- namedtuple(): 生成可以使用名字来访问元素内容的tuple子类- 
- deque: 双端队列，可以快速的从另外一侧追加和推出对象
- Counter: 计数器，主要用来计数
- OrderedDict: 有序字典
- defaultdict: 带有默认值的字典

### 双端队列，deque的用法(其他的以后有用到再看)
deque其实是 double-ended queue 的缩写，翻译过来就是双端队列，它最大的好处就是实现了从队列 头部快速增加和取出对象: .popleft(), .appendleft() 。

但是值得注意的是，list对象的这两种用法的时间复杂度是 O(n) ，也就是说随着元素数量的增加耗时呈 线性上升。而使用deque对象则是 O(1) 的复杂度，所以当你的代码有这样的需求的时候， 一定要记得使用deque。
```
import collections

d = collections.deque('abcdefg')
print 'Deque:',d
print 'Length:',len(d)
print 'Left end:',d[0]
print 'Right end:'d[-1]

- **Add to the right**

d1 = collections.deque()
d1.extend('abcdefg')
d1.append('h')
print 'append: d1

- **Add to the left**

d2 = collections.deque()
d2.extendleft(xrange(6))
print 'extendleft:', d2
d2.appendleft(6)
print 'appendleft:', d2

还有更多功能。。。自己百度去
```
**举个栗子**
```
# -*- coding: utf-8 -*-
```
下面这个是一个有趣的例子，主要使用了deque的rotate方法来实现了一个无限循环
的加载动画
```
```
import sys
import time
from collections import deque

fancy_loading = deque('>--------------------')

while True:
    print '\r%s' % ''.join(fancy_loading),
    fancy_loading.rotate(1)
    sys.stdout.flush()
    time.sleep(0.08)
```


## Jupyter notebook的使用
### linux安装简单:

> `sudo pip install jupyter notebook `

启动jupyer notebook，在终端输入:

> `jupyter notebook`

会自动打开默认浏览器名为locallhost:8888/tree

### [怎样远程访问jupyter notebook](http://blog.csdn.net/bitboy_star/article/details/51427306)
- 1、生成配置文件
```
$ jupyter notebook --generate-config
```
- 2、生成密码
打开ipython,创建一个密文的密码：
```
In [1]: from notebook.auth import passwd
In [2]: passwd()
Enter password:
Verify password:
Out[2]:'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'
```
把生成的密文'xxxxxxxxxxxxxxxxxxxxxxx'复制保存
- 3、修改默认配置文件
```
$vim ~/.jupyter/jupyter_notebook_config.py 
```
进行如下修改：
```
c.NotebookApp.ip='*'
c.NotebookApp.password = u'sha:ce...刚才复制的那个密文'
c.NotebookApp.open_browser = False
c.NotebookApp.port =8888 #随便指定一个端口
```
- 4、启动jupyter notebook

`$ jupyter notebook`

- 5、设置后台运行
```
jupyter notebook &> /dev/null &
```
用top加kill可以关闭，但也可以配置快捷命令
```
$ echo 'alias quitjupyter="kill $(pgrep jupyter)"' >> ~/.bashrc
```
```
$ quitjupyter
```
jupyter notebook是一个强大的工具，更为强大的命令以后再学吧

