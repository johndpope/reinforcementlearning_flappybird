# -*- coding: utf-8 -*-
# file: main.py
# author: JinTian
# time: 22/06/2017 11:06 AM
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
from keras import backend as K
import tensorflow as tf
from models import build_model, build_model_mine
import gym
import game.wrapped_flappy_bird as game
import numpy as np
from global_config import *
import matplotlib.pyplot as plt
from PIL import Image
from collections import deque
import random
import json
import os


def process_image(img):
    # input array output array with some process, converts to gray and resize it
    image = Image.fromarray(img)
    print(np.array(image).shape)
    image = image.convert('L')
    image = image.resize((128, 72))
    image = np.array(image)
    image = np.expand_dims(image, axis=2)
    print(image.shape)
    return image


def get_next_state(action, game_state):
    frame0, reward0, dead = game_state.frame_step(action)
    frame0 = process_image(frame0)
    return frame0, reward0, dead


def get_next_action(t, state_t, deterministic_model):
    # get t+1 time action, state_t is actually is the neural network input, in this case it is an image
    action_t = np.zeros(ACTIONS)
    if t % ACTION_PER_FRAME == 0:
        # a probability to choose random, another predict using model
        print('generate next action, using prob {}'.format(RANDOM_EXPLORE_PROB))
        if np.random.rand() < RANDOM_EXPLORE_PROB:
            print('-- action from random explore --')
            action_index = np.random.randint(ACTIONS)
            action_t[action_index] = 1
        else:
            print('-- action from experience --')
            state_t = np.expand_dims(state_t, axis=0)
            predict = deterministic_model.predict(state_t)
            predict = np.array(predict).squeeze(axis=0)
            # get the reward most action
            action_t[np.argmax(predict)] = 1
    return action_t


def predict_goals(model, state):
    # using state, which is an image or a array
    state = np.expand_dims(state, axis=0)
    predict = model.predict(state)
    # convert predict to 1-dim
    predict = np.array(predict).squeeze(axis=0)
    return predict


def main():
    game_state = game.GameState()

    model = build_model_mine()
    if os.path.exists('mine_model.h5'):
        model.load_weights('mine_model.h5')

    # all scenes experienced by Agent. we using deque() to contain it, consider it as list if you like
    # every scene contains states which is image frame, and the rewards of different actions
    # etc. frame{(72, 128, 1)} -> goals{[0.1, -1, 0.3, 0.4]}, goal dim is same as actions, cause it indicates
    # every action accordingly reward.
    all_scenes = deque()
    loss = 0

    init_action = np.zeros(ACTIONS)
    init_action[0] = 1
    state_t, _, _ = get_next_state(init_action, game_state)

    t = 0
    observation_holder = NUM_OBSERVATIONS
    while True:
        if t < observation_holder:
            print('======= keep observation.')
            action_t_next = get_next_action(t, state_t, model)

            # next state will run once game, and get the state_next which is an image
            state_t_next, reward_next, dead = get_next_state(action_t_next, game_state)

            goal_experience = predict_goals(model, state_t)
            goal = predict_goals(model, state_t_next)
            if dead:
                goal[np.argmax(action_t_next)] = reward_next
            else:
                goal[np.argmax(action_t_next)] = reward_next + GAMA * np.argmax(goal_experience)

            scene = (state_t, goal)
            all_scenes.append(scene)

            state_t = state_t_next
            print('gathered a scene.')

        else:
            print('======= observation stop, start train model from all scenes.')
            batch_size = 32
            # this sample will continue running when observation finished.
            scenes = random.sample(all_scenes, batch_size)

            # states is all batch of frame images which is state
            # goals is the accordingly rewards in terms of state and after different actions
            states = []
            goals = []
            for i in range(len(scenes)):
                states.append(scenes[i][0])
                goals.append(scenes[i][1])
            states = np.array(states)
            goals = np.array(goals)
            # train on collected states and goals
            l = model.train_on_batch(states, goals)
            loss += l
            if t % 1000 == 0:
                model.save_weights('mine_model.h5', overwrite=True)
                with open('mine_model.json', 'w') as f:
                    json.dump(model.to_json(), f)
                print('model and weights were saved at time: ', t)
                print('observe another round!')
                observation_holder = t + NUM_OBSERVATIONS
        t += 1
        print('solving {}'.format(t))


def dummy_play():
    def random_action():
        action = np.zeros(2)
        action[np.random.randint(2)] = 1
        return action
    # dummy play using random action to see what happens
    game_state = game.GameState()
    action_t = np.zeros(2)
    action_t[0] = 1
    while True:
        a = random_action()
        print('get random action: ', a)
        frame, r, dead = game_state.frame_step(a)
        # original frame image shape is (288, 512, 3) => (72, 128, 1) resize and gray scale
        print(f'frame: {frame.shape}, reward: {r}, dead: {dead}')
        if dead:
            print('game over.')
            break


if __name__ == "__main__":
    # dummy_play()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    K.set_session(sess)
    main()
