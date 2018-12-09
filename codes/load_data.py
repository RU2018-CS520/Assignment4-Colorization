#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 23:34:00 2018

@author: zx123
"""

#import cPickle as pickle
import pickle
import numpy as np

def load_cifar_train_data():
    '''
    train_images format = (index, channel, height, width)
    train_labels format = (index, label)
    '''
    train_images = []
    train_labels = np.array([])

    for iter in range(1, 6):
        batch_file_name = 'cifar-10-batches-py/data_batch_' + str(iter)
        print('Importing ' +  batch_file_name)
        batch_file = open(batch_file_name, 'rb')
        dict = pickle.load(batch_file, encoding='bytes')
        batch_file.close()

        batch_images = dict[b'data']
        batch_labels = dict[b'labels']

        train_images.append(batch_images)
        train_labels = np.append(train_labels, batch_labels)

    train_images = np.vstack(train_images) / 256.0
    train_images = train_images.reshape(train_images.shape[0], 3, 32, 32)

    return (train_images, train_labels)


def load_cifar_val_data():
    '''
    val_images format = (index, channel, height, width)
    val_labels format = (index, label)
    '''

    file_name = 'cifar-10-batches-py/test_batch'
    file = open(file_name, 'rb')
    dict = pickle.load(file, encoding='bytes')
    file.close()

    val_images = dict[b'data'] / 256.0
    val_images = val_images.reshape(val_images.shape[0], 3, 32, 32)

    val_labels = np.array(dict[b'labels'])

    return (val_images, val_labels)
