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


array = [[9,1,0,0,0],  # when input was A, prediction was A for 9 times, B for 1 time
         [1,15,3,1,3], # when input was B, prediction was A for 1 time, B for 15 times, C for 3 times
         [5,0,24,1,4], # when input was C, prediction was A for 5 times, C for 24 times, D for 1 time
         [3,0,4,25,1],
         [0,4,1,15,4]] # when input was D, prediction was B for 4 times, C for 1 time, D for 15 timess
# df_cm = pd.DataFrame(array, index = [i for i in "ABCD"],
#                   columns = [i for i in "ABCD"])
# plt.figure(figsize = (10,7))
# plt.title('confusion matrix without normalization')
# sns.heatmap(df_cm, annot=True)

len = 5

total = np.sum(array, axis = 1)
for i in range(len):
    for j in range(len):
        array[i][j] = float(array[i][j]) / total[i]


df_cm = pd.DataFrame(array, index = [i for i in "ABCDE"],
                  columns = [i for i in "ABCDE"])
plt.figure(figsize = (10,7))
plt.title('confusion matrix with normalization')
sns.heatmap(df_cm, annot=True)

plt.show()