# Lab 7 Learning rate and Evaluation
from __future__ import print_function
from sklearn.metrics import confusion_matrix
import tensorflow as tf
import random
import numpy as np

# import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import sys

tf.set_random_seed(777)  # reproducibility

# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# Check out https://www.tensorflow.org/get_started/mnist/beginners for
# more information about the mnist dataset


# xy = np.loadtxt('/home/rainman/datasets/mydata/20171101/step2_2/171101_2.csv', delimiter=',', dtype=np.float32)
xy = np.loadtxt('/home/rainman/git/emotionClassifer/emotion_classifier/2_MakeRandomCSV/out/20171121_train.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1]
y_raw = xy[:, [-1]]
y_raw = np.reshape(y_raw,xy.shape[0])
y_raw = np.int_(y_raw)

# for one_hot code,
y_data = np.zeros((xy.shape[0], 5))
y_data[np.arange(xy.shape[0]), y_raw] = 1

print('num of parameter : ', xy.shape[1]-1)

# indices = [0, 2, -1, 1]
# depth = 3
# on_value = 5.0
# off_value = 0.0
# axis = -1
#
# out = tf.one_hot(indices = indices, depth=depth, on_value=on_value, off_value=off_value, axis=-1)
# print(out)
xy2 = np.loadtxt('/home/rainman/git/emotionClassifer/emotion_classifier/2_MakeRandomCSV/out/20171121_train.csv', delimiter=',', dtype=np.float32)
x_data2 = xy2[:, 0:-1]
y_raw = xy2[:, [-1]]
y_raw = np.reshape(y_raw,xy2.shape[0])
y_raw = np.int_(y_raw)

# for one_hot code,
y_data2 = np.zeros((xy2.shape[0], 5))
y_data2[np.arange(xy2.shape[0]), y_raw] = 1
# parameters
learning_rate = 0.01
training_epochs = 100
num_examples = xy.shape[0]
batch_size = 100

# input place holders
X = tf.placeholder(tf.float32, [None, xy.shape[1]-1])
Y = tf.placeholder(tf.float32, [None, 5])

# # weights & bias for nn layers
# W = tf.Variable(tf.random_normal([6, 5]))
# b = tf.Variable(tf.random_normal([5]))
#
# hypothesis = tf.matmul(X, W) + b

keep_prob = tf.placeholder(tf.float32)
W1 = tf.get_variable("W1", shape=[xy.shape[1]-1, 5],
                     initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.random_normal([5]))

w1_hist = tf.summary.histogram("weights1", W1)
b1_hist = tf.summary.histogram("biases1", b1)

# L1 = tf.nn.relu(tf.matmul(X, W1) + b1)
# L1 = tf.nn.dropout(L1, keep_prob=keep_prob)
#
# W2 = tf.get_variable("W2", shape=[5, 5],
#                      initializer=tf.contrib.layers.xavier_initializer())
# b2 = tf.Variable(tf.random_normal([5]))
# L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)
# L2 = tf.nn.dropout(L2, keep_prob=keep_prob)
#
# W3 = tf.get_variable("W3", shape=[5, 5],
#                      initializer=tf.contrib.layers.xavier_initializer())
# b3 = tf.Variable(tf.random_normal([5]))
# L3 = tf.nn.relu(tf.matmul(L2, W3) + b3)
# L3 = tf.nn.dropout(L3, keep_prob=keep_prob)
#
# W4 = tf.get_variable("W4", shape=[5, 5],
#                      initializer=tf.contrib.layers.xavier_initializer())
# b4 = tf.Variable(tf.random_normal([5]))
# L4 = tf.nn.relu(tf.matmul(L3, W4) + b4)
# L4 = tf.nn.dropout(L4, keep_prob=keep_prob)
#
# W5 = tf.get_variable("W5", shape=[5, 5],
#                      initializer=tf.contrib.layers.xavier_initializer())
# b5 = tf.Variable(tf.random_normal([5]))
hypothesis = tf.matmul(X, W1) + b1

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
writer = tf.summary.FileWriter("./logs/single")
writer.add_graph(sess.graph)  # Show the graph
# train my model
global_step=0
for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(num_examples / batch_size)

    for i in range(total_batch):

        # batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        # feed_dict = {X: batch_xs, Y: batch_ys}
        summary, c, _, acc = sess.run([merged_summary, cost, optimizer, accuracy], feed_dict={X: x_data, Y: y_data, keep_prob: 0.7})
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
print('Accuracy:', sess.run(accuracy, feed_dict={X: x_data2, Y: y_data2, keep_prob: 0.7}))


# make confusion matrix
orig_stdout = sys.stdout
f = open('out.txt', 'w')
sys.stdout = f

print("label    prediction")
array = [[0, 0, 0, 0, 0],  # when input was A, prediction was A for 9 times, B for 1 time
         [0, 0, 0, 0, 0], # when input was B, prediction was A for 1 time, B for 15 times, C for 3 times
         [0, 0, 0, 0, 0], # when input was C, prediction was A for 5 times, C for 24 times, D for 1 time
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0]]



for r in range(100):    # r 0= random.randint(0, xy2.shape[0] - 1)
    l = sess.run(tf.argmax(y_data2[r:r+1], 1))
    p = sess.run(
        tf.argmax(hypothesis, 1), feed_dict={X: x_data2[r:r + 1]})

    array[l[0]][p[0]]+=1
    # print(l, p)

# print results

for i in range(5):
    for j in range(5):
        print (array[i][j], end=',')
    print("")

sys.stdout = orig_stdout
f.close()

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
