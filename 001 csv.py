# Read the dataset files in 'datasets' directory using relative path.
# Read the lines (rows) in the csv and split the string into a label and image data.
# Convert the image data which is in 1D vector form into 2D vector using 'for' loops.
# Create the 'outputs/train' and 'outputs/test' directories, if not exists.
# Save each individual 2D vector image into '#{label}-#{row_index}.csv' under train or test outputs directory.

import os, glob
import numpy as np
import csv

path_to_dataset_1 = os.path.join('datasets', 'mnist_train.csv')
path_to_dataset_2 = os.path.join('datasets', 'mnist_train_2.csv')
path_to_dataset_test = os.path.join('datasets', 'mnist_test.csv')

read_dataset_2 = csv.reader(open(path_to_dataset_2, 'r'))
label_train = np.empty()
a = 0
for row in read_dataset_2:
    data = row[1:]
    label = row[0]
    data_2d = np.reshape(data, (28, 28))
    output_file = "001 csv/outputs/train_2/" + "{}-{}.csv".format(label, a)
    csv.writer(open(output_file, 'w', newline = '')).writerows(data_2d)
    a += 1
print(a)


read_dataset_1 = csv.reader(open(path_to_dataset_1, 'r'))
b = 0
for row in read_dataset_1:
    data = row[1:]
    label = row[0]
    data_2d = np.reshape(data, (28, 28))
    output_file = "001 csv/outputs/train/" + "{}-{}.csv".format(label, b)
    csv.writer(open(output_file, 'w', newline = '')).writerows(data_2d)
    b += 1
print(b)

read_dataset_test = csv.reader(open(path_to_dataset_test, 'r'))
c = 0
for row in read_dataset_test:
    data = row[1:]
    label = row[0]
    data_2d = np.reshape(data, (28, 28))
    output_file = "001 csv/outputs/test/" + "{}-{}.csv".format(label, c)
    csv.writer(open(output_file, 'w', newline = '')).writerows(data_2d)
    c += 1