# Lab 7 Learning rate and Evaluation
from __future__ import print_function
from datetime import timedelta

import tensorflow as tf
import math
import random
import numpy as np
import matplotlib.pyplot as plt

import os
import errno
import sys
from mpl_toolkits.mplot3d import Axes3D
PLOT_DIR = './out'
SORT_OF_EMO = 6
SORT_OF_APPTYPE = 5
# OUTMODE = 'show'
OUTMODE = 'none'
OUT_FILE_NUM = 10

class NODE:

    x_idx = []
    y_idx = []
    z_idx = []
    s = []
    c = []
node = NODE()
deleted = []


def qSort(left, right, arr, x, y, z):
    i=left
    j=right
    p=arr[(i+j)/2]

    while(i<=j) :
        while(arr[i] > p) :
            i+=1
        while (arr[j] < p):
            j-=1
        if(i<=j):
            tmp = arr[i]
            arr[i] = arr[j]
            arr[j] = tmp

            tmp = x[i]
            x[i] = x[j]
            x[j] = tmp

            tmp = y[i]
            y[i] = y[j]
            y[j] = tmp

            tmp = z[i]
            z[i] = z[j]
            z[j] = tmp

            i+=1
            j-=1
    if(i<right):
        qSort(i,right,arr, x,y,z)
    if(left<j):
        qSort(left, j, arr, x,y,z)


def show3dGraph(x_2dim_2d, h):
    x = np.arange(0, SORT_OF_EMO, 1)
    y = np.arange(0, SORT_OF_EMO, 1)
    X, Y = np.meshgrid(x, y)

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_surface(X, Y, x_2dim_2d)

    plot_dir = os.path.join(PLOT_DIR, 'graph')

    # create directory if does not exist, otherwise empty it
    if not os.path.exists(plot_dir):
        try:
            os.makedirs(plot_dir)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise

    plt.savefig(os.path.join(plot_dir, '{}.png'.format('3d{}'.format(h) )), bbox_inches='tight')
    if OUTMODE is 'show' :
        plt.show()
    return

def setNodeValues(x_2dim, h) :

    node.s = [i * 3000 for i in x_2dim[h]]
    node.c = []
    qSort(0, 215, node.s, node.x_idx, node.y_idx, node.z_idx)
    cnt = 0
    diff = []
    pre = 0
    totalDiff = 0
    for i in node.s:
        if cnt > 0 :
            diff.append(pre - i)

        if (cnt < 1):
            node.c.append('r')
        elif (cnt < 5):
            node.c.append('b')
        elif (cnt < 20):
            node.c.append('y')
        else:
            node.c.append('black')
        if(cnt > 0 and cnt < 6) :
            totalDiff += (pre - i)

        cnt += 1
        pre = i
    totalDiff /= 5;
    if totalDiff <10 :
        for j in range (0,216):
            x_2dim[h][j] = 0.0
            # node.s[j] = 0.0
        deleted.append(1)
    else :
        deleted.append(0)

    return


def show4dScatter(x_2dim, h):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(node.x_idx, node.y_idx, node.z_idx, s=node.s, c=node.c, cmap=plt.hot())

    plot_dir = os.path.join(PLOT_DIR, 'scatter')

    # create directory if does not exist, otherwise empty it
    if not os.path.exists(plot_dir):
        try:
            os.makedirs(plot_dir)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise

    if(deleted[h] == 1) :
        plt.savefig(os.path.join(plot_dir, '{}.png'.format('4d{}_deleted'.format(h))), bbox_inches='tight')
    else :
        plt.savefig(os.path.join(plot_dir, '{}.png'.format('4d{}'.format(h) )), bbox_inches='tight')

    if OUTMODE is 'show' :
        plt.show()
    return


tf.set_random_seed(777)  # reproducibility
xy = np.loadtxt('/home/rainman/git/emotionClassifer/emotion_classifier/2_MakeRandomCSV/out/train0611_40s.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1]
x_2dim = []

for i in range(len(x_data[1])) :
    for j in range(len(x_data[1])):
        for k in range(len(x_data[1])):
            node.x_idx.append(i)
            node.y_idx.append(j)
            node.z_idx.append(k)

# under 0.1 return
for h in range(xy.shape[0]):
    x_2dim.append([])
#6*6*6
    for i in range(len(x_data[1])) :
        if (x_data[h][i] < 0.01) :
            x_data[h][i] = 0

        for j in range(len(x_data[1])):
            if (x_data[h][j] < 0.01):
                x_data[h][j] = 0

            for k in range(len(x_data[1])):
                if (x_data[h][k] < 0.01):
                    x_data[h][k] = 0
                x_2dim[h].append(x_data[h][i]*x_data[h][j]*x_data[h][k])

    if (h < OUT_FILE_NUM):
        setNodeValues(x_2dim, h)
        show4dScatter(x_2dim, h)


y_raw = xy[:, [-1]]
y_raw = np.reshape(y_raw,xy.shape[0])
y_raw = np.int_(y_raw)

# for one_hot code,
y_data = np.zeros((xy.shape[0], SORT_OF_APPTYPE))
y_data[np.arange(xy.shape[0]), y_raw] = 1

print('num of parameter : ', xy.shape[1]-1)

xy2 = np.loadtxt('/home/rainman/git/emotionClassifer/emotion_classifier/2_MakeRandomCSV/out/test0611_40s.csv', delimiter=',', dtype=np.float32)

x_data2 = xy2[:, 0:-1]

x_2dim2 = []

# under 0.1 return
for h in range(xy2.shape[0]):
    x_2dim2.append([])
#6*6*6
    for i in range(len(x_data2[1])) :
        if (x_data2[h][i] < 0.01) :
            x_data2[h][i] = 0

        for j in range(len(x_data2[1])):
            if (x_data2[h][j] < 0.01):
                x_data2[h][j] = 0

            for k in range(len(x_data2[1])):
                if (x_data2[h][k] < 0.01):
                    x_data2[h][k] = 0
                # for m in range(len(x_data2[1])):
                #     if (x_data2[h][m] < 0.01):
                #         x_data2[h][m] = 0

                x_2dim2[h].append(x_data2[h][i] * x_data2[h][j] *x_data2[h][k])

y_raw = xy2[:, [-1]]
y_raw = np.reshape(y_raw,xy2.shape[0])
y_raw = np.int_(y_raw)

# for one_hot code,
y_data2 = np.zeros((xy2.shape[0], SORT_OF_APPTYPE))
y_data2[np.arange(xy2.shape[0]), y_raw] = 1

# parameters
learning_rate = 0.01
training_epochs = 100
num_examples = xy.shape[0]
batch_size = 100

WH = xy.shape[1]-1
INPUTSIZE =WH*WH*WH

# input place holders
X = tf.placeholder(tf.float32, [None, INPUTSIZE],  name='X')
Y = tf.placeholder(tf.float32, [None, SORT_OF_APPTYPE],  name='Y')

keep_prob = tf.placeholder(tf.float32)
W1 = tf.get_variable("W1", shape=[INPUTSIZE, INPUTSIZE*8],
                     initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.random_normal([INPUTSIZE*8]))

L1 = tf.nn.relu(tf.matmul(X, W1) + b1)
L1 = tf.nn.dropout(L1, keep_prob=keep_prob)
#
W2 = tf.get_variable("W2", shape=[INPUTSIZE*8, INPUTSIZE],
                     initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.random_normal([INPUTSIZE]))
L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)
L2 = tf.nn.dropout(L2, keep_prob=keep_prob)

W3 = tf.get_variable("W3", shape=[INPUTSIZE, SORT_OF_APPTYPE],
                     initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.random_normal([SORT_OF_APPTYPE]))

hypothesis = tf.matmul(L2, W3) + b3

# # Optimization.
# regularization_loss = 0.5 * tf.reduce_sum(tf.square(W3))
# hinge_loss = tf.reduce_sum(tf.maximum(tf.zeros([5000, SORT_OF_APPTYPE]),
#                                       1 - Y * hypothesis))
# cost = regularization_loss + 1 * hinge_loss
# optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(cost)


# define cost/loss & optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=hypothesis, labels=Y))
cost_summ = tf.summary.scalar("cost", cost)

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
accuracy_summ = tf.summary.scalar("accuracy", accuracy)

# initialize
sess = tf.Session()
sess.run(tf.global_variables_initializer())
merged_summary = tf.summary.merge_all()
writer = tf.summary.FileWriter("./logs")
writer.add_graph(sess.graph)  # Show the graph
# train my model
global_step=0
for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(num_examples / batch_size)

    for i in range(total_batch):

        summary, c, _, acc = sess.run([merged_summary, cost, optimizer, accuracy], feed_dict={X: x_2dim, Y: y_data, keep_prob: 0.7})
        avg_cost = c
        # writer.add_summary(c, 1)

    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))
    print('Accuracy at step %s: %s' % (epoch, acc))
    writer.add_summary(summary, global_step=global_step)
    global_step += 1

print('Learning Finished!')

# Test model and check accuracy
correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print('Accuracy:', sess.run(accuracy, feed_dict={X: x_2dim2, Y: y_data2, keep_prob: 0.7}))

# make confusion matrix
d = timedelta(microseconds=-1)

orig_stdout = sys.stdout
ofilename = 'out_mul'+str(d.days) +str(d.seconds) +'.txt'
f = open( ofilename, 'w')
sys.stdout = f

print("label    prediction")

#test
array = [[0, 0, 0, 0, 0],  # when input was A, prediction was A for 9 times, B for 1 time
         [0, 0, 0, 0, 0], # when input was B, prediction was A for 1 time, B for 15 times, C for 3 times
         [0, 0, 0, 0, 0], # when input was C, prediction was A for 5 times, C for 24 times, D for 1 time
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0]]


data_np = np.asarray(x_2dim2, np.float32)

for r in range(xy2.shape[0]):    # r = random.randint(0, xy2.shape[0] - 1)
    l = sess.run(tf.argmax(y_data2[r:r+1], 1))

    # p_tf = tf.convert_to_tensor(x_2dim2, np.float32)
    p = sess.run(
        tf.argmax(hypothesis, 1), feed_dict={X: data_np[r:r+1],  keep_prob: 0.7} )
    array[l[0]][p[0]]+=1
    print(l, p)

for i in range(5):
    for j in range(5):
        print (array[i][j], end=',')
    print("")


#training
array = [[0, 0, 0, 0, 0],  # when input was A, prediction was A for 9 times, B for 1 time
         [0, 0, 0, 0, 0], # when input was B, prediction was A for 1 time, B for 15 times, C for 3 times
         [0, 0, 0, 0, 0], # when input was C, prediction was A for 5 times, C for 24 times, D for 1 time
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0]]
data_np = np.asarray(x_2dim, np.float32)

for r in range(xy.shape[0]):    # r = random.randint(0, xy2.shape[0] - 1)
    l = sess.run(tf.argmax(y_data[r:r+1], 1))

    # p_tf = tf.convert_to_tensor(x_2dim2, np.float32)
    p = sess.run(
        tf.argmax(hypothesis, 1), feed_dict={X: data_np[r:r+1],  keep_prob: 0.7} )
    array[l[0]][p[0]]+=1
    print(l, p)

for i in range(5):
    for j in range(5):
        print (array[i][j], end=',')
    print("")

# validation
array = [[0, 0, 0, 0, 0],  # when input was A, prediction was A for 9 times, B for 1 time <
         [0, 0, 0, 0, 0],  # when input was B, prediction was A for 1 time, B for 15 times, C for 3 times
         [0, 0, 0, 0, 0],  # when input was C, prediction was A for 5 times, C for 24 times, D for 1 time
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0]]
data_np = np.asarray(x_2dim, np.float32)


capa = [200,200,200,200,200]

r = 0
while (r < xy.shape[0]) :
# for r in range(xy.shape[0]):  # r = random.randint(0, xy2.shape[0] - 1)
    l = sess.run(tf.argmax(y_data[r:r + 1], 1))
    if capa[l[0]] <= 0 :
        r += (xy.shape[0] / 5)
        continue
    capa[l[0]] -= 1

    # p_tf = tf.convert_to_tensor(x_2dim2, np.float32)
    p = sess.run(
        tf.argmax(hypothesis, 1), feed_dict={X: data_np[r:r + 1], keep_prob: 0.7})
    array[l[0]][p[0]] += 1
    print(l, p)
    r += 1

for i in range(5):
    for j in range(5):
        print(array[i][j], end=',')
    print("")

# # Get one and predict
# r = random.randint(0, mnist.test.num_examples - 1)
# print("Label: ", sess.run(tf.argmax(mnist.test.labels[r:r + 1], 1)))
# print("Prediction: ", sess.run(
#     tf.argmax(hypothesis, 1), feed_dict={X: mnist.test.images[r:r + 1]}))

# plt.imshow(mnist.test.images[r:r + 1].
#           reshape(28, 28), cmap='Greys', interpolation='nearest')
# plt.show()

'''
Epoch: 0001 cost = 5.888845987
Epoch: 0002 cost = 1.860620173
Epoch: 0003 cost = 1.159035648
Epoch: 0004 cost = 0.892340870
Epoch: 0005 cost = 0.751155428
Epoch: 0006 cost = 0.662484806
Epoch: 0007 cost = 0.601544010
Epoch: 0008 cost = 0.556526115
Epoch: 0009 cost = 0.521186961
Epoch: 0010 cost = 0.493068354
Epoch: 0011 cost = 0.469686249
Epoch: 0012 cost = 0.449967254
Epoch: 0013 cost = 0.433519321
Epoch: 0014 cost = 0.419000337
Epoch: 0015 cost = 0.406490815
Learning Finished!
Accuracy: 0.9035
'''
