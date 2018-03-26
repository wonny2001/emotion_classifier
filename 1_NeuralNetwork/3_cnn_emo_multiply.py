# Lab 7 Learning rate and Evaluation
import tensorflow as tf
import math
import random
import numpy as np
# import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
tf.set_random_seed(777)  # reproducibility

# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# Check out https://www.tensorflow.org/get_started/mnist/beginners for
# more information about the mnist dataset

xy = np.loadtxt('/home/rainman/git/emotionClassifer/emotion_classifier/2_MakeRandomCSV/out/20171121_train.csv', delimiter=',', dtype=np.float32)
# xy = np.loadtxt('171008.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1]
SIZE_Y = 5
x_2dim = []

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

                for m in range(len(x_data[1])):
                    if (x_data[h][m] < 0.01):
                        x_data[h][m] = 0
                    x_2dim[h].append(x_data[h][i]*x_data[h][j]*x_data[h][k]*x_data[h][m])


y_raw = xy[:, [-1]]
y_raw = np.reshape(y_raw,xy.shape[0])
y_raw = np.int_(y_raw)

# for one_hot code,
y_data = np.zeros((xy.shape[0], SIZE_Y))
y_data[np.arange(xy.shape[0]), y_raw] = 1

print('num of parameter : ', xy.shape[1]-1)

xy2 = np.loadtxt('/home/rainman/git/emotionClassifer/emotion_classifier/2_MakeRandomCSV/out/20171121_test.csv', delimiter=',', dtype=np.float32)

# xy2 = np.loadtxt('171008.csv', delimiter=',', dtype=np.float32)
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

                for m in range(len(x_data2[1])):
                    if (x_data2[h][m] < 0.01):
                        x_data2[h][m] = 0

                    x_2dim2[h].append(x_data2[h][i]*x_data2[h][k]*x_data2[h][m])


y_raw = xy2[:, [-1]]
y_raw = np.reshape(y_raw,xy2.shape[0])
y_raw = np.int_(y_raw)

# for one_hot code,
y_data2 = np.zeros((xy2.shape[0], SIZE_Y))
y_data2[np.arange(xy2.shape[0]), y_raw] = 1
# parameters
learning_rate = 0.01
training_epochs = 100
num_examples = xy.shape[0]
batch_size = 100

WH = xy.shape[1]-1
WH = WH*WH
INPUTSIZE =WH*WH
THIRD_WH = 9

###########################


# input place holders
X = tf.placeholder(tf.float32, [None, INPUTSIZE])
X_img = tf.reshape(X, [-1, WH, WH, 1])   # img 28x28x1 (black/white)
Y = tf.placeholder(tf.float32, [None, SIZE_Y])

# L1 ImgIn shape=(?, 28, 28, 1)
W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))
#    Conv     -> (?, 28, 28, 32)
#    Pool     -> (?, 14, 14, 32)
L1 = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding='SAME')
L1 = tf.nn.relu(L1)
L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1],
                    strides=[1, 2, 2, 1], padding='SAME')
'''
Tensor("Conv2D:0", shape=(?, 28, 28, 32), dtype=float32)
Tensor("Relu:0", shape=(?, 28, 28, 32), dtype=float32)
Tensor("MaxPool:0", shape=(?, 14, 14, 32), dtype=float32)
'''

# L2 ImgIn shape=(?, 14, 14, 32)
W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
#    Conv      ->(?, 14, 14, 64)
#    Pool      ->(?, 7, 7, 64)
L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1],
                    strides=[1, 2, 2, 1], padding='SAME')
L2_flat = tf.reshape(L2, [-1, THIRD_WH * THIRD_WH * 64])
'''
Tensor("Conv2D_1:0", shape=(?, 14, 14, 64), dtype=float32)
Tensor("Relu_1:0", shape=(?, 14, 14, 64), dtype=float32)
Tensor("MaxPool_1:0", shape=(?, 7, 7, 64), dtype=float32)
Tensor("Reshape_1:0", shape=(?, 3136), dtype=float32)
'''

# Final FC 7x7x64 inputs -> 10 outputs
W3 = tf.get_variable("W3", shape=[THIRD_WH * THIRD_WH * 64, SIZE_Y],
                     initializer=tf.contrib.layers.xavier_initializer())
b = tf.Variable(tf.random_normal([SIZE_Y]))
logits = tf.matmul(L2_flat, W3) + b

###########################


##################
#
# # input place holders
# X = tf.placeholder(tf.float32, [None, INPUTSIZE])
# Y = tf.placeholder(tf.float32, [None, 5])
#
# keep_prob = tf.placeholder(tf.float32)
# W1 = tf.get_variable("W1", shape=[INPUTSIZE, INPUTSIZE*8],
#                      initializer=tf.contrib.layers.xavier_initializer())
# b1 = tf.Variable(tf.random_normal([INPUTSIZE*8]))
#
# L1 = tf.nn.relu(tf.matmul(X, W1) + b1)
# L1 = tf.nn.dropout(L1, keep_prob=keep_prob)
# #
# W2 = tf.get_variable("W2", shape=[INPUTSIZE*8, INPUTSIZE],
#                      initializer=tf.contrib.layers.xavier_initializer())
# b2 = tf.Variable(tf.random_normal([INPUTSIZE]))
# L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)
# L2 = tf.nn.dropout(L2, keep_prob=keep_prob)
#
# W3 = tf.get_variable("W3", shape=[INPUTSIZE, 5],
#                      initializer=tf.contrib.layers.xavier_initializer())
# b3 = tf.Variable(tf.random_normal([5]))
#
# hypothesis = tf.matmul(L2, W3) + b3

#######################33

# define cost/loss & optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
cost_summ = tf.summary.scalar("cost", cost)

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
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

        # batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        # feed_dict = {X: batch_xs, Y: batch_ys}
        summary, c, _, acc = sess.run([merged_summary, cost, optimizer, accuracy], feed_dict={X: x_2dim, Y: y_data})
        avg_cost = c
        # writer.add_summary(c, 1)

    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))
    print('Accuracy at step %s: %s' % (epoch, acc))
    writer.add_summary(summary, global_step=global_step)
    global_step += 1

print('Learning Finished!')

# Test model and check accuracy
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print('Accuracy:', sess.run(accuracy, feed_dict={X: x_2dim2, Y: y_data2}))

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
