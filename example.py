import cv2
import numpy as np
import os
from random import shuffle

import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

import matplotlib.pyplot as plt


train_dir = 'F:/Fun with python/ML/CatsnDogs/train'
test_dir = 'F:/Fun with python/ML/CatsnDogs/test'
img_size = 50
LR = 1e-3

MODEL_NAME = 'dogsvscats-{}-{}.model'.format(LR, '6conv-basic-15epoch')

def label_img(img):
    word_label = img.split('.')[-3]
    if word_label == 'cat': return [1,0]
    elif word_label == 'dog': return [0,1]

def create_train_data():
    training_data = []
    for img in os.listdir(train_dir):
        label = label_img(img)
        path = os.path.join(train_dir, img)
        img = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (img_size, img_size))
        training_data.append([np.array(img), np.array(label)])
    shuffle(training_data)
    np.save('train_data.npy', training_data)
    return training_data

def process_test_data():
    testing_data = []
    for img in os.listdir(test_dir):
        path = os.path.join(test_dir, img)
        img_num = img.split('.')[0]
        img = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (img_size, img_size))
        testing_data.append([np.array(img), img_num])

    np.save('test_data.npy', testing_data)
    return testing_data


convnet = input_data(shape=[None,img_size,img_size,1], name='input')

convnet = conv_2d(convnet, 32, 2, activation='relu')
convnet = max_pool_2d(convnet,2)

convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet,2)

convnet = conv_2d(convnet, 32, 2, activation='relu')
convnet = max_pool_2d(convnet,2)

convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet,2)

convnet = conv_2d(convnet, 32, 2, activation='relu')
convnet = max_pool_2d(convnet,2)

convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet,2)

convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet,0.8)

convnet = fully_connected(convnet,2,activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=LR,loss='categorical_crossentropy',name='targets')

model = tflearn.DNN(convnet, tensorboard_dir='log')


# print(os.path.exists('dogsvscats-0.001-6conv-basic-15epoch.model.meta'))
#testing
##test_data = process_test_data()
test_data = np.load('test_data.npy')
#####

if os.path.exists(MODEL_NAME+'.meta'):
    model.load(MODEL_NAME)
    print("Model Loaded!")

##else:
##    train_data = create_train_data()
##    train = train_data[:-500]
##    test = train_data[-500:]
##    
##    X = np.array([i[0] for i in train]).reshape(-1, img_size, img_size, 1)
##    Y = [i[1] for i in train]
##
##    test_x = np.array([i[0] for i in test]).reshape(-1, img_size, img_size, 1)
##    test_y = [i[1] for i in test]
##
##    model.fit({'input': X}, {'targets': Y}, n_epoch=15,
##              validation_set=({'input': test_x}, {'targets': test_y}),
##              snapshot_step=500, show_metric=True, run_id=MODEL_NAME)

#testing
fig = plt.figure()
for num, data in enumerate(test_data[:12]):
    #cat:[1,0]
    #dog:[0,1]

    img_num = data[1]
    img_data = data[0]

    y = fig.add_subplot(3,4,num+1)
    orig = img_data
    data = img_data.reshape(img_size, img_size, 1)

    model_out = model.predict([data])[0]

    if np.argmax(model_out) == 1:
        str_label = 'Dog'
    else:
        str_label = 'Cat'

    y.imshow(orig, cmap='gray')
    plt.title(str_label)
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)

plt.show()

with open('submission_f', 'w') as f:
    f.write('id,label\n')

with open('submission_f', 'a') as f:
    i = 0
    for data in test_data:
        print(i)
        img_num = data[1]
        img_data = data[0]
        orig = img_data
        data = img_data.reshape(img_size, img_size, 1)
        model_out = model.predict([data])[0]
        f.write('{},{}\n'.format(img_num, model_out[1]))
        i += 1
