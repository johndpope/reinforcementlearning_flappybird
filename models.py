# -*- coding: utf-8 -*-
# file: models.py
# author: JinTian
# time: 22/06/2017 11:07 AM
# Copyright 2017 JinTian. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------
import json
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.convolutional import Conv2D
from keras.optimizers import SGD, Adam

from global_config import *


def build_model():
    print("Now we build the model")
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(8, 8), strides=(4, 4), padding='same',
                     input_shape=IMAGE_SHAPE))
    model.add(Activation('relu'))
    model.add(Conv2D(64, kernel_size=(4, 4), strides=(2, 2), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same'))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(2))

    adam = Adam(lr=LR)
    model.compile(loss='mse', optimizer=adam)
    print("We finish building the model")
    return model


def build_model_mine():
    print("-- building mine model...")
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(8, 8), strides=(4, 4), padding='same',
                     input_shape=IMAGE_SHAPE))
    model.add(Activation('relu'))
    model.add(Conv2D(64, kernel_size=(4, 4), strides=(2, 2), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same'))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(2))

    adam = Adam(lr=LR)
    model.compile(loss='mse', optimizer=adam)
    print("-- building model finished.")
    return model
