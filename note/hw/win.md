真地没有想过会在windows上装tensorflow，一直都在把程序跑在ubuntu系统上的，自己在学校上一直都用ssh或远程桌面的方式操作运行程序。这次回家本来计划的好好的，不过一回到家学校的机器就不知怎么地连不上了，心中一万只草泥马在奔腾。。。

一开始有想过申请个云服务器的，不过呢，看了看网上的价格，真是让人退而却步。我就想着先用自己的电脑试试tensorflow的CPU版本，观察一下运行速度怎样，结果发现这个版本的速度真心的慢。如果租了一个便宜的云服务器，估计也是一样的慢速度，所以我打算还是用我的带有笔记本的电脑装tensorflow的GPU版本。在Windows上安装tensorflow也走了一些弯路，一开始不知道是否必须要安装cuda,才能安装tensorflow;tensorflow是否支持pip原生安装等。最后通过半天的摸索，终于可以跑GPU了。在笔记本电脑上路tensorflow其实不那么爽，听着风扇在呼呼的转，可想电脑的负荷有多大。我还是喜欢把写程序的工作区与跑程序的工作区分开。下面记录一下tensorflow在Win10上安装tensorflow的一些成功方法。
## 1.安装cuda和cudnn
windows版本先安装cuda，exe安装很方便，不过要满足一样的条件，比如要先安装vs2013等，cudnn复制到cuda的目录下。
## 2.pip安装
- **CPU版本安装**
安装的最好教程还是参考tensorflow的官网，安装指令如下：
```
pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/windows/cpu/tensorflow-0.12.0rc1-cp35-cp35m-win_amd64.whl
```
这里的关键就是`--ignorre-installed`，不加的话会为一个错误过不去。

- **GPU版本安装**

```
pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/windows/gpu/tensorflow_gpu-0.12.0rc1-cp35-cp35m-win_amd64.whl
```
**注**：win10的PowerShell的开启与把反应速度很慢，不知是我电脑的问题还是系统的问题，所以一开始是在PowerShell装的，因为慢就想用cmd的打开，结果不能用呀。所以最后卸了conda env，直接改为在cmd中pip安装。
## 3.conda安装

从这次的安装尝试，我才知道conda流行的原因，它也是一种强大的python包管理工具，具有同virtualenv的功能。下面只介绍GPU版本的安装，其实它的安装只有多加了前面的conda环境的控制 

```
conda create -n tensorflow-gpu python=3.5 # 建立一个虚拟环境
activate tensorflow
pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/windows/gpu/tensorflow_gpu-0.12.0rc1-cp35-cp35m-win_amd64.whl
```