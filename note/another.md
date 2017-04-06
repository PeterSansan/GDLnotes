## 第二章:Softmax分类函数
```python
# -*- coding:utf-8 -*-

"""Example Softmax:Udacity class deep learning"""

import numpy as np
# scores = [3.0,1,0,0.2]
scores = np.array([[1,2,3,6],
					[2,4,5,6],
					[3,8,7,6]])

def softmax(x):
	"""Compute softmax values for x."""
	return np.exp(x)/np.sum(np.exp(x),axis=0)# 注意这里是沿着行求和
	
print(softmax(scores))

# Plot softmax curves
import matplotlib.pyplot as plt
x = np.arange(-2.0,6.0,0.1)
scores = np.vstack([x,np.ones_like(x),0.2* np.ones_like(x)]) # 垂直组合，变成三行，x在不断变大，另外两行固定

plt.plot(x,softmax(scores).T,linewidth=2)
plt.show()
```
结果：
![result](http://ogtxggxo6.bkt.clouddn.com/softmax.png?imageslim)

**如果都把结果放大10位**
![img](http://ogtxggxo6.bkt.clouddn.com/softmax2.png?imageslim)

**如果把结果都除以10**

![img](http://ogtxggxo6.bkt.clouddn.com/softmax3.png?imageslim)

### 结论： 上面的结果就是Softmax的输入数值越大，分类区分度越大，输入数值越小，分类区分度越小。

第三章：NotMNIST实验
**注意**：NotMNIST数据集官方给的代码下载不下来，可以换一个,使用迅雷下下来超快
```
http://yaroslavvb.com/upload/notMNIST/notMNIST_large.tar.gz
http://yaroslavvb.com/upload/notMNIST/notMNIST_small.tar.gz
```
- ### 问题1：显示原始数据
```
from IPython.display import Image
Image(filename='./A/ZXRjaHkudHRm.png')
```
或者
```
from IPython.display import display, Image
display(Image(filename="notMNIST_small/A/Q0NXaWxkV29yZHMtQm9sZEl0YWxpYy50dGY=.png"))
```
![img](http://ogtxggxo6.bkt.clouddn.com/aa.png?imageslim)

- ### 问题2：查看归一化零均值，同方差的图片
```
with open('A.pickle', 'rb') as pk_f:
    data4show = pickle.load(pk_f)
print(data4show.shape)
pic = data4show[0,:,:]
plt.imshow(pic)
plt.show()

# (52909,28,28)
```
![img](http://ogtxggxo6.bkt.clouddn.com/aaa.png?imageslim)
- ### 问题3：验证各个类的数目是否均衡
```
# count numbers in different classes
file_path = 'notMNIST_large/{0}.pickle'
for ele in 'ABCDEFJHIJ':
    with open(file_path.format(ele), 'rb') as pk_f:
        dat = pickle.load(pk_f)
    print('number of pictures in {}.pickle = '.format(ele), dat.shape[0])
```
```
number of pictures in A.pickle =  52909
number of pictures in B.pickle =  52911
number of pictures in C.pickle =  52912
number of pictures in D.pickle =  52911
number of pictures in E.pickle =  52912
number of pictures in F.pickle =  52912
number of pictures in J.pickle =  52911
number of pictures in H.pickle =  52912
number of pictures in I.pickle =  52912
number of pictures in J.pickle =  52911
```
- ### 检测数据是否在随想打乱后依然完好
```
mapping = {key:val for key, val in enumerate('ABCDEFGHIJ')}
def plot_check(matrix, key):
    plt.imshow(matrix)
    plt.show()
    print('the picture should be ', mapping[key])
    return None

length = train_dataset.shape[0]-1
# five check
for _ in xrange(5):
    index = np.random.randint(length)
    plot_check(train_dataset[index,:,:], train_labels[index])
```
后面的我就没看了，详细可以对照参考[这里](http://www.itnose.net/detail/6617671.html)

## 第三章：随机梯度下降法与随机梯度下降法（SGD）
两者都采用相同的优化函数`tf.train.GradientDescentOptimizer`,但是后者多了batch_size和使用placeholder

后面在这个基础上再加上一层relu的隐藏层
```
# coding: utf-8

# Deep Learning
# =============
# Udacity上的课程
# Assignment 2：NotMinst全连接的神经网络分类程序（简单）,在这基础上再添加一层relu的隐藏层
# ------------
from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range



pickle_file = 'notMNIST.pickle'

with open(pickle_file, 'rb') as f:
  save = pickle.load(f)
  train_dataset = save['train_dataset']
  train_labels = save['train_labels']
  valid_dataset = save['valid_dataset']
  valid_labels = save['valid_labels']
  test_dataset = save['test_dataset']
  test_labels = save['test_labels']
  del save  # hint to help gc free up memory
  print('Training set', train_dataset.shape, train_labels.shape)  	# [200000,28,28]
  print('Validation set', valid_dataset.shape, valid_labels.shape) 	# [10000,28,28]
  print('Test set', test_dataset.shape, test_labels.shape)			# [10000,28,28]


# Reformat into a shape that's more adapted to the models we're going to train:
# - data as a flat matrix,
# - labels as float 1-hot encodings.

image_size = 28  
num_labels = 10   

def reformat(dataset, labels):
  dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
  # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  return dataset, labels
train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)


train_subset = 10000  # 训练集的个数

graph = tf.Graph()
with graph.as_default():

  # Input data.
  # Load the training, validation and test data into constants that are
  # attached to the graph.
  tf_train_dataset = tf.constant(train_dataset[:train_subset, :])
  tf_train_labels = tf.constant(train_labels[:train_subset])
  tf_valid_dataset = tf.constant(valid_dataset)
  tf_test_dataset = tf.constant(test_dataset)
  
  # Variables.
  # These are the parameters that we are going to be training. The weight
  # matrix will be initialized using random values following a (truncated)
  # normal distribution. The biases get initialized to zero.
  weights = tf.Variable(
    tf.truncated_normal([image_size * image_size, num_labels]))
  biases = tf.Variable(tf.zeros([num_labels]))
  
  # Training computation.

  logits = tf.matmul(tf_train_dataset, weights) + biases  # 这里是把数据一次性输入运算，这就是梯度下降的做法													# 这里是把全部数据都输入，展示梯度下降算法的做法
  loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))
  
  # Optimizer.
  # We are going to find the minimum of this loss using gradient descent.
  optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
  
  # Predictions for the training, validation, and test data.
  # These are not part of training, but merely here so that we can report
  # accuracy figures as we train.
  train_prediction = tf.nn.softmax(logits)
  valid_prediction = tf.nn.softmax(
    tf.matmul(tf_valid_dataset, weights) + biases)
  test_prediction = tf.nn.softmax(tf.matmul(tf_test_dataset, weights) + biases)



num_steps = 801

def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

with tf.Session(graph=graph) as session:
  # This is a one-time operation which ensures the parameters get initialized as
  # we described in the graph: random weights for the matrix, zeros for the
  # biases. 
  tf.global_variables_initializer().run()
  print('Initialized')
  for step in range(num_steps):
  
    _, l, predictions = session.run([optimizer, loss, train_prediction])
    if (step % 100 == 0):
      print('Loss at step %d: %f' % (step, l))
      print('Training accuracy: %.1f%%' % accuracy(
        predictions, train_labels[:train_subset, :]))

      print('Validation accuracy: %.1f%%' % accuracy(
        valid_prediction.eval(), valid_labels))
  print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))


  
# 下面是随机梯度下降的方法，使用了batch_size与placeholder 
# Let's now switch to stochastic gradient descent training instead, which is much faster.
# 
# The graph will be similar, except that instead of holding all the training data into a constant node, we create a `Placeholder` node which will be fed actual data at every call of `session.run()`.


batch_size = 128
hidden_size = 64

graph = tf.Graph()
with graph.as_default():

  # Input data. For the training data, we use a placeholder that will be fed
  # at run time with a training minibatch.
  tf_train_dataset = tf.placeholder(tf.float32,
                                    shape=(batch_size, image_size * image_size))
  tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
  tf_valid_dataset = tf.constant(valid_dataset)
  tf_test_dataset = tf.constant(test_dataset)
  
  
  weights0 = tf.Variable(tf.truncated_normal([image_size * image_size, hidden_size])) # [24*24,hidden_size]
  biases0 = tf.Variable(tf.zeros([hidden_size]))     # [hidden_size]
  
  # 隐藏层1
  hidden1 = tf.nn.relu(tf.matmul(tf_train_dataset, weights0) + biases0)

  # softmax层
  weights1 = tf.Variable(tf.truncated_normal([hidden_size,num_labels]))
  biases1 = tf.Variable(tf.zeros([num_labels]))
  
  logits = tf.matmul(hidden1,weights1)+biases1
  
  loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))
  
  # Optimizer.
  optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
  
  # Predictions for the training, validation, and test data.
  train_prediction = tf.nn.softmax(logits)
  valid_prediction = tf.nn.softmax(tf.matmul(tf.nn.relu(tf.matmul(tf_valid_dataset, weights0)+biases0),weights1)+biases1)
  test_prediction = tf.nn.softmax(tf.matmul(tf.nn.relu(tf.matmul(tf_test_dataset, weights0)+biases0),weights1)+biases1)


# Let's run it:

num_steps = 12001

with tf.Session(graph=graph) as session:
  tf.global_variables_initializer().run()
  print("Initialized")
  for step in range(num_steps):
    # Pick an offset within the training data, which has been randomized.
    # Note: we could use better randomization across epochs.
    offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
    # Generate a minibatch.
    batch_data = train_dataset[offset:(offset + batch_size), :]
    batch_labels = train_labels[offset:(offset + batch_size), :]
    # Prepare a dictionary telling the session where to feed the minibatch.
    # The key of the dictionary is the placeholder node of the graph to be fed,
    # and the value is the numpy array to feed to it.
    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
    _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
    if (step % 500 == 0):
      print("Minibatch loss at step %d: %f" % (step, l))
      print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
      print("Validation accuracy: %.1f%%" % accuracy(
        valid_prediction.eval(), valid_labels))
  print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))

# ---
# Problem
# -------
# 
# Turn the logistic regression example with SGD into a 1-hidden layer neural network with rectified linear units [nn.relu()](https://www.tensorflow.org/versions/r0.7/api_docs/python/nn.html#relu) and 1024 hidden nodes. This model should improve your validation / test accuracy.
# 
# ---
```
## 第四章：防止过拟合的方法
除了改变或增加数据集外，还有几种提高泛化能力常见的方法：**正则化**、**Dropout**；在下面练习中加入了L2正则化、Droput、学习速率衰减的方法
```

# coding: utf-8

# Deep Learning
# =============
# 
# Assignment 3：The goal of this assignment is to explore regularization techniques.
# ------------


from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle



pickle_file = 'notMNIST.pickle'

with open(pickle_file, 'rb') as f:
  save = pickle.load(f)
  train_dataset = save['train_dataset']
  train_labels = save['train_labels']
  valid_dataset = save['valid_dataset']
  valid_labels = save['valid_labels']
  test_dataset = save['test_dataset']
  test_labels = save['test_labels']
  del save  # hint to help gc free up memory
  print('Training set', train_dataset.shape, train_labels.shape)
  print('Validation set', valid_dataset.shape, valid_labels.shape)
  print('Test set', test_dataset.shape, test_labels.shape)


# Reformat into a shape that's more adapted to the models we're going to train:
# - data as a flat matrix,
# - labels as float 1-hot encodings.

image_size = 28
num_labels = 10

def reformat(dataset, labels):
  dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
  # Map 1 to [0.0, 1.0, 0.0 ...], 2 to [0.0, 0.0, 1.0 ...]
  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  return dataset, labels
train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)


def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

# 建立模型，SGD
batch_size = 128
hidden_size = 256
regularation_param = 0.001
prob=0.5

graph = tf.Graph()
with graph.as_default():

  # Input data. For the training data, we use a placeholder that will be fed
  # at run time with a training minibatch.
  tf_train_dataset = tf.placeholder(tf.float32,
                                    shape=(batch_size, image_size * image_size))
  tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
  tf_valid_dataset = tf.constant(valid_dataset)
  tf_test_dataset = tf.constant(test_dataset)
  
  
  weights0 = tf.Variable(tf.truncated_normal([image_size * image_size, hidden_size])) # [24*24,hidden_size]
  biases0 = tf.Variable(tf.zeros([hidden_size]))     # [hidden_size]
  
  # 隐藏层1
  hidden1 = tf.nn.relu(tf.matmul(tf_train_dataset, weights0) + biases0)
  # 问题3：dropout层
  keep_prob = tf.placeholder("float")
  h_fc1_drop = tf.nn.dropout(hidden1, keep_prob)
  
  # softmax层
  weights1 = tf.Variable(tf.truncated_normal([hidden_size,num_labels]))
  biases1 = tf.Variable(tf.zeros([num_labels]))
  
  logits = tf.matmul(h_fc1_drop,weights1)+biases1

  # 没有正则化
  loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))
  # 问题1：L2正则化
  # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits)+regularation_param*tf.nn.l2_loss(weights0)+tf.nn.l2_loss(weights1))
  
  # 问题4：学习速率下降
  # Optimizer.
  decay_rate = 0.96
  decay_steps = 10000
  global_step = tf.Variable(0)  # count the number of steps taken.
  learning_rate = tf.train.exponential_decay(0.5, global_step,decay_steps,decay_rate) # 指数下降
  optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

  # optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
  # Predictions for the training, validation, and test data.
  train_prediction = tf.nn.softmax(logits)
  valid_prediction = tf.nn.softmax(tf.matmul(tf.nn.relu(tf.matmul(tf_valid_dataset, weights0)+biases0),weights1)+biases1)
  test_prediction = tf.nn.softmax(tf.matmul(tf.nn.relu(tf.matmul(tf_test_dataset, weights0)+biases0),weights1)+biases1)

# 运行程序

num_steps = 50001

with tf.Session(graph=graph) as session:
  tf.global_variables_initializer().run()
  print("Initialized")
  for step in range(num_steps):
    # Pick an offset within the training data, which has been randomized.
    # Note: we could use better randomization across epochs.
    offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
    # Generate a minibatch.
    batch_data = train_dataset[offset:(offset + batch_size), :]
    batch_labels = train_labels[offset:(offset + batch_size), :]
    # Prepare a dictionary telling the session where to feed the minibatch.
    # The key of the dictionary is the placeholder node of the graph to be fed,
    # and the value is the numpy array to feed to it.
    feed_dict_train = {tf_train_dataset : batch_data, tf_train_labels : batch_labels , keep_prob : prob}
    _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict_train)
    if (step % 500 == 0):
      print("Minibatch loss at step %d: %f" % (step, l))
      print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
      print("Validation accuracy: %.1f%%" % accuracy(valid_prediction.eval(), valid_labels))
  print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))

		  
# ---
# Problem 1
# ---------
# 
# Introduce and tune L2 regularization for both logistic and neural network models. Remember that L2 amounts to adding a penalty on the norm of the weights to the loss. In TensorFlow, you can compute the L2 loss for a tensor `t` using `nn.l2_loss(t)`. The right amount of regularization should improve your validation / test accuracy.
# 
# ---

# ---
# Problem 2
# ---------
# Let's demonstrate an extreme case of overfitting. Restrict your training data to just a few batches. What happens?
# 
# ---

# ---
# Problem 3
# ---------
# Introduce Dropout on the hidden layer of the neural network. Remember: Dropout should only be introduced during training, not evaluation, otherwise your evaluation results would be stochastic as well. TensorFlow provides `nn.dropout()` for that, but you have to make sure it's only inserted during training.
# 
# What happens to our extreme overfitting case?
# 
# ---

# ---
# Problem 4
# ---------
# 
# Try to get the best performance you can using a multi-layer model! The best reported test accuracy using a deep network is [97.1%](http://yaroslavvb.blogspot.com/2011/09/notmnist-dataset.html?showComment=1391023266211#c8758720086795711595).
# 
# One avenue you can explore is to add multiple layers.
# 
# Another one is to use learning rate decay:
# 
#     global_step = tf.Variable(0)  # count the number of steps taken.
#     learning_rate = tf.train.exponential_decay(0.5, global_step, ...)
#     optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
#  
#  ---
# 
```
## 第五章：卷积神经网络
[这里贴出了视频中的图片](http://blog.csdn.net/myjiayan/article/details/52155075)
![img](http://ogtxggxo6.bkt.clouddn.com/cc2.png?imageslim)
![img](http://ogtxggxo6.bkt.clouddn.com/cnn1.png?imageslim)
![img](http://ogtxggxo6.bkt.clouddn.com/cnn5.png?imageslim)
![img](http://ogtxggxo6.bkt.clouddn.com/cnn7.png?imageslim)
![img](http://ogtxggxo6.bkt.clouddn.com/cnn9.png?imageslim)
![img](http://ogtxggxo6.bkt.clouddn.com/cnn10.png?imageslim)
![img](http://ogtxggxo6.bkt.clouddn.com/cnn8.png?imageslim)
![img](http://ogtxggxo6.bkt.clouddn.com/cnn6.png?imageslim)
![img](http://ogtxggxo6.bkt.clouddn.com/cnn4.png?imageslim)
