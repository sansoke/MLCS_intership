#==========================================#
# Title:  Splitting mnist dataset into train, validation, and test sets
# Author: Jaewoong Han modified by Jeanho Kim
# Date:   2025-01-28
#==========================================#


import os
import pathlib
import glob
import numpy as np
import random

files = glob.glob('001 csv/outputs/**/*/*.csv', recursive = True)

mnist_2d = np.empty((len(files), 28, 28))
labels = np.empty(len(files))

for file_idx, file_path in enumerate(files):
    data = np.loadtxt(file_path, delimiter = ",")
    for i in range(28):
        for j in range(28):
            mnist_2d[file_idx, i, j] = data[i][j]
    labels[file_idx] = os.path.basename(file_path).split('-')[0]

label_data = np.array(list(zip(labels, mnist_2d)), dtype = object)
random.shuffle(label_data)

train_data = int(len(files)*0.7)
val_data = int(len(files)*0.9)

print(train_data, val_data)


train, val, test = np.split(label_data, [train_data, val_data])

train_x, train_y = zip(*train)
val_x, val_y = zip(*val)
test_x, test_y = zip(*test)

np.savez("train.npz", data = train_x, label = train_y)
np.savez("val.npz", data = val_x, label = val_y)
np.savez("test.npz", data = test_x, label = test_y)