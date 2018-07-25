from datetime import timedelta
from operator import add
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
sns.set(font_scale=2)
import numpy as np
from numpy import linalg as LA

# array = [[5,0,0,0],  # when input was A, prediction was all A
#          [0,10,0,0], # when input was B, prediction was all A
#          [0,0,15,0], # when input was C, prediction was all A
#          [0,0,0,5]]  # when input was D, prediction was all A
# df_cm = pd.DataFrame(array, index = [i for i in "ABCD"],
#                   columns = [i for i in "ABCD"])
# plt.figure(figsize = (10,7))
# plt.title('confusion matrix without confusion')
# sns.heatmap(df_cm, annot=True)
d = timedelta(microseconds=-1)
ofilename = 'out_mul'+str(d.days) +str(d.seconds) +'.txt'
len = 5


arrayTest = [[192,7,7,13,0],
         [4,204,0,2,10],
         [3,0,218,0,0],
         [19,1,2,193,0],
         [0,3,0,0,215]]

arrayTrain = [[2144,19,25,10,1],
         [12,2182,1,3,6],
         [6,1,2204,2,0],
         [38,10,8,2102,0],
         [0,3,0,0,2186]]

arrayValid = [[214,2,3,1,0],
         [1,218,0,0,1],
         [1,0,220,0,0],
         [4,1,1,210,0],
         [0,1,0,0,218]]

array = [[0, 0, 0, 0, 0],  # when input was A, prediction was A for 9 times, B for 1 time
         [0, 0, 0, 0, 0], # when input was B, prediction was A for 1 time, B for 15 times, C for 3 times
         [0, 0, 0, 0, 0], # when input was C, prediction was A for 5 times, C for 24 times, D for 1 time
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0]]

for i in range(len):
    for j in range(len):
        array[i][j] = arrayTest[i][j] + arrayTrain[i][j] + arrayValid[i][j]
# array = [[794,39,32,8,127],
#          [7,674,277,1,41],
#          [4,65,861,8,62],
#          [5,0,131,846,18],
#          [55,52,206,96,591]]
# df_cm = pd.DataFrame(array, index = [i for i in "ABCD"],
#                   columns = [i for i in "ABCD"])
# plt.figure(figsize = (10,7))
# plt.title('confusion matrix without normalization')
# sns.heatmap(df_cm, annot=True)


sum_line = [0,0,0,0,0]
sum_col = [0,0,0,0,0]
recall = [0,0,0,0,0]
precision = [0,0,0,0,0]
for i in range(len):
    for j in range(len):
       sum_line[i] += array[i][j]

for j in range(len):
    for i in range(len):
       sum_col[j] += array[i][j]



for i in range(len):
    recall[i] = float(array[i][i]) / sum_col[i]
    print('recall ', i, ':', recall[i])


for i in range(len):
    precision[i] = float(array[i][i]) / sum_line[i]
    print('precision ', i, ':', precision[i])


total = np.sum(array, axis = 1)
totalAll = np.sum(array)
for i in range(len):
    for j in range(len):
        array[i][j] = float(array[i][j]) / totalAll


COL = ['Clock', 'Gallery', 'SNS', 'Game', 'Official']

df_cm = pd.DataFrame(array , index = [i for i in COL],
                  columns = [i for i in COL])
plt.figure(figsize = (10,7))
# plt.title('Test Confusion matrix')
# plt.title('Train Confusion matrix')
plt.title('Validation Confusion matrix')
sns.heatmap(df_cm, annot=True)

plt.show()