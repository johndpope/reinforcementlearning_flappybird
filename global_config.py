# -*- coding: utf-8 -*-
# file: global_config.py
# author: JinTian
# time: 22/06/2017 11:10 AM
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


# =========== model options =========
IMAGE_SHAPE = (72, 128, 1)
LR = 0.001


# =========== game options ==========
# actions are up and do nothing, it decided on questions
ACTIONS = 2
ACTION_PER_FRAME = 1
# not train model, just observation gather the experience.
NUM_OBSERVATIONS = 600
# probability of random explore
RANDOM_EXPLORE_PROB = 0.1

# past experience weight
GAMA = 0.3



