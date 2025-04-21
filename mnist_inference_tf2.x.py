'''This is inference code for mnist dataset '''

from __future__ import print_function
import tensorflow.keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os
import tensorflow as tf
import pandas  #수정

# Recreate the exact same model, including its weights and the optimizer
model = tf.keras.models.load_model('saved_model/')

# Show the model architecture
model.summary()

##-- Model Test using Test datasets
print()
print("----Actual test for digits----")

mnist_label_file_path = 't_labels.txt'
mnist_label = open(mnist_label_file_path, "r")
cnt_correct = 0
for index in range(10):
	#-- read a label
	label = mnist_label.readline().strip()  #수정

	#-- formatting the input image (image data)
	img = Image.open('dataset_test/testimgs/' + str(index+1) + '.png').convert("L")
	img = img.resize((28,28))
	im2arr = np.array(img).reshape(1,28,28,1)
	im2arr = im2arr / 255.0  #수정

	# Predicting the Test set results
	y_prob = model.predict(im2arr)          #수정
	y_pred = y_prob.argmax(axis=-1)[0]      #수정

	print()
	print("label = {} --> predicted label= {}".format(label, y_pred))

	#-- compute the accuracy of the preditcion
	if int(label)==y_pred:  #수정
		cnt_correct += 1

#-- Final accuracy
Final_acc = cnt_correct/10
print()
print("Final test accuracy: %f" %Final_acc)
print()
print('****tensorflow version****:',tf.__version__)
print()

data = {
    '이름': ['이연우'],
    '학번': [2411861],
    '학과': ['인공지능공학부']
}

df = pandas.DataFrame(data)
print(df)
