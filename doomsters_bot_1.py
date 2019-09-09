#!/usr/bin/env python
# -*- coding: utf-8 -*-

# +-+-+-+-+-+-+-+ +-+-+-+ +-+-+-+-+
# |V|I|Z|D|O|O|M| |B|O|T| |2|0|1|9|
# +-+-+-+-+-+-+-+ +-+-+-+ +-+-+-+-+

from __future__ import division
from __future__ import print_function
import itertools as it
from random import sample, randint, random
from time import time, sleep
import numpy as np
import skimage.color, skimage.transform
import tensorflow as tf
from tqdm import trange
import vizdoom as vzd
from argparse import ArgumentParser
import sys
import logging

# +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+

# - Variable definition


# +-+-+-+-+-+-+-+ +-+-+-+-+-+-+-+-+-+-+
# |C|L|A|S|S|E|S| |D|E|F|I|N|I|T|I|O|N|
# +-+-+-+-+-+-+-+ +-+-+-+-+-+-+-+-+-+-+

class AISetup:
    internalLearningRate = 0.00025
    discountFactor = 0.99
    learningStepsPerEpochs = 2000
    epochs = 30
    replayMemorySize = 10000
    batchSize = 64
    testEpisodesPerEpoch = 100
    frameRepeat = 12
    resolution = (30,45)
    episodesToWatch = 10
    bots = 0
    saveModel = False
    loadModel = True
    skipLearning = True
    defaultModelFolder = ""
    defaultConfiguration = ""
    trainingList = (True,True,False)
    competitionList = (False,True,True)

    def setDefaultModelFolder( self, folder ):
        self.defaultModelFolder = folder

    def setDefaultConfiguration( self, configuration ):
        self.defaultConfiguration = configuration

    def setMode( self, mode ):
        if mode == 'TRAINING':
            self.saveModel,self.loadModel,self.skipLearning = self.trainingList
        elif mode == 'COMPETITION':
            self.saveModel,self.loadModel,self.skipLearning = self.competitionList

    def preprocess(img):
        NominalRange = (0.25, 0.5)
        #myTempShape = img.shape
        #print( myTempShape )
        #transform2 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        transform2 = img
        myTempShape2 = img.shape
        #print( myTempShape2 )
        #cv2.imwrite('gray.png',transform2)
        myAlto, myLargo = myTempShape2
        newAlto = int( myAlto * NominalRange[0] )
        newLargo = int( myLargo * NominalRange[1] )
        #print( newAlto )
        #print( newLargo )
        tempImagex = transform2[newAlto:newLargo,:]
        #cv2.imwrite('crop.png',tempImagex)
        #img = skimage.transform.resize(img, resolution)
        #cv2.imwrite('final1.png',img)
        #img = img.astype(np.float32)
        #cv2.imwrite('final2.png',img)
        #return img
        blur = cv2.GaussianBlur(tempImagex,(5,5),0)
        # find normalized_histogram, and its cumulative distribution function
        hist = cv2.calcHist([blur],[0],None,[256],[0,256])
        hist_norm = hist.ravel()/hist.max()
        Q = hist_norm.cumsum()
        bins = np.arange(256)
        fn_min = np.inf
        thresh = -1
        for i in range(1,256):
            p1,p2 = np.hsplit(hist_norm,[i]) # probabilities
            q1,q2 = Q[i],Q[255]-Q[i] # cum sum of classes
            b1,b2 = np.hsplit(bins,[i]) # weights
            #finding means and variances
            m1,m2 = np.sum(p1*b1)/q1, np.sum(p2*b2)/q2
            v1,v2 = np.sum(((b1-m1)**2)*p1)/q1,np.sum(((b2-m2)**2)*p2)/q2
            # calculates the minimization function
            fn = v1*q1 + v2*q2
            if fn < fn_min:
                fn_min = fn
                thresh = i
            # find otsu's threshold value with OpenCV function
            ret, otsu = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            #print thresh,ret
            #cv2.imwrite('otsu.png',otsu)
            #print( otsu )
            return otsu

    def create_network(session, available_actions_count):
        # Create the input variables
        s1_ = tf.placeholder(tf.float32, [None] + list(resolution) + [1], name="State")
        a_ = tf.placeholder(tf.int32, [None], name="Action")
        target_q_ = tf.placeholder(tf.float32, [None, available_actions_count], name="TargetQ")

        # Add 2 convolutional layers with ReLu activation
        conv1 = tf.contrib.layers.convolution2d(s1_, num_outputs=8, kernel_size=[6, 6], stride=[3, 3],
                                            activation_fn=tf.nn.relu,
                                            weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                            biases_initializer=tf.constant_initializer(0.1))
        conv2 = tf.contrib.layers.convolution2d(conv1, num_outputs=8, kernel_size=[3, 3], stride=[2, 2],
                                            activation_fn=tf.nn.relu,
                                            weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                            biases_initializer=tf.constant_initializer(0.1))
        conv2_flat = tf.contrib.layers.flatten(conv2)
        fc1 = tf.contrib.layers.fully_connected(conv2_flat, num_outputs=128, activation_fn=tf.nn.relu,
                                            weights_initializer=tf.contrib.layers.xavier_initializer(),
                                            biases_initializer=tf.constant_initializer(0.1))

        q = tf.contrib.layers.fully_connected(fc1, num_outputs=available_actions_count, activation_fn=None,
                                          weights_initializer=tf.contrib.layers.xavier_initializer(),
                                          biases_initializer=tf.constant_initializer(0.1))
        best_a = tf.argmax(q, 1)
        loss = tf.losses.mean_squared_error(q, target_q_)

        optimizer = tf.train.RMSPropOptimizer(learning_rate)
        # Update the parameters according to the computed gradient using RMSProp.
        train_step = optimizer.minimize(loss)

        def function_learn(s1, target_q):
            feed_dict = {s1_: s1, target_q_: target_q}
            l, _ = session.run([loss, train_step], feed_dict=feed_dict)
            return l

        def function_get_q_values(state):
            return session.run(q, feed_dict={s1_: state})

        def function_get_best_action(state):
            return session.run(best_a, feed_dict={s1_: state})

        def function_simple_get_best_action(state):
            return function_get_best_action(state.reshape([1, resolution[0], resolution[1], 1]))[0]

        return function_learn, function_get_q_values, function_simple_get_best_action


    def learn_from_memory():
        """ Learns from a single transition (making use of replay memory).
        s2 is ignored if s2_isterminal """

        # Get a random minibatch from the replay memory and learns from it.
        if memory.size > batch_size:
            s1, a, s2, isterminal, r = memory.get_sample(batch_size)

            q2 = np.max(get_q_values(s2), axis=1)
            target_q = get_q_values(s1)
            # target differs from q only for the selected action. The following means:
            # target_Q(s,a) = r + gamma * max Q(s2,_) if not isterminal else r
            target_q[np.arange(target_q.shape[0]), a] = r + discount_factor * (1 - isterminal) * q2
            learn(s1, target_q)

    def perform_learning_step(epoch):
        """ Makes an action according to eps-greedy policy, observes the result
        (next state, reward) and learns from the transition"""

        def exploration_rate(epoch):
            """# Define exploration rate change over time"""
            start_eps = 1.0
            end_eps = 0.1
            const_eps_epochs = 0.1 * epochs  # 10% of learning time
            eps_decay_epochs = 0.6 * epochs  # 60% of learning time

            if epoch < const_eps_epochs:
                return start_eps
            elif epoch < eps_decay_epochs:
                # Linear decay
                return start_eps - (epoch - const_eps_epochs) / \
                               (eps_decay_epochs - const_eps_epochs) * (start_eps - end_eps)
            else:
                return end_eps

        s1 = preprocess(game.get_state().screen_buffer)

        # With probability eps make a random action.
        eps = exploration_rate(epoch)
        if random() <= eps:
            a = randint(0, len(actions) - 1)
        else:
            # Choose the best action according to the network.
            a = get_best_action(s1)
        reward = game.make_action(actions[a], frame_repeat)

        isterminal = game.is_episode_finished()
        s2 = preprocess(game.get_state().screen_buffer) if not isterminal else None

        # Remember the transition that was just experienced.
        memory.add_transition(s1, a, s2, isterminal, reward)

        learn_from_memory()

# +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+

class ReplayMemory:
    def __init__(self, capacity):
        channels = 1
        state_shape = (capacity, resolution[0], resolution[1], channels)
        self.s1 = np.zeros(state_shape, dtype=np.float32)
        self.s2 = np.zeros(state_shape, dtype=np.float32)
        self.a = np.zeros(capacity, dtype=np.int32)
        self.r = np.zeros(capacity, dtype=np.float32)
        self.isterminal = np.zeros(capacity, dtype=np.float32)

        self.capacity = capacity
        self.size = 0
        self.pos = 0

    def add_transition(self, s1, action, s2, isterminal, reward):
        self.s1[self.pos, :, :, 0] = s1
        self.a[self.pos] = action
        if not isterminal:
            self.s2[self.pos, :, :, 0] = s2
        self.isterminal[self.pos] = isterminal
        self.r[self.pos] = reward

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def get_sample(self, sample_size):
        i = sample(range(0, self.size), sample_size)
        return self.s1[i], self.a[i], self.s2[i], self.isterminal[i], self.r[i]

# +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+

class playerBehaviour:

    # . Explore
    move_forward = [0,0,0,0,1,0,0]
    move_backward = [0,0,0,1,0,0,0]
    move_forward_turning_right = [0,0,0,0,1,1,0]
    move_forward_turning_left = [0,0,0,0,1,0,1]

    # . Defense
    move_backward_turning_right = [0,0,0,1,0,1,0]
    move_backward_turning_left = [0,0,0,1,0,0,1]
    attack_and_move_backward = [1,0,0,1,0,0,0]
    move_backward_turning_right_and_attacking = [1,0,0,1,0,1,0]

    # . Attack
    move_forward_and_turning_left_and_attacking = [1,0,0,0,1,0,1]
    attack = [1,0,0,0,0,0,0]
    attack_and_move_forward = [1,0,0,0,1,0,0]

    # . Groups
    explore = [move_forward,move_backward,move_forward_turning_right,move_forward_turning_left]
    defence = [move_backward_turning_right,move_backward_turning_left,attack_and_move_backward,move_backward_turning_right_and_attacking]
    attack = [attack,move_forward_and_turning_left_and_attacking,attack_and_move_forward]

    # . Final action pack
    actions = []
    def playerBehaviour(self,playerBehaviourFlag):
        if playerBehaviourFlag == 1:
            # . set agressive mode
            for i in range(1000):
                tempList = sample(self.attack,1)
                #print(tempList[0])
                self.actions.append(tempList[0])
            #print( self.actions )
        elif playerBehaviourFlag == 2:
            # . set defence mode
            for i in range(1000):
                tempList = sample(self.defence,1)
                self.actions.append(tempList[0])


        elif playerBehaviourFlag == 3:
            # . combined behaviour
           for i in range(1000):
                tempList = sample(self.defence,1)
                tempList2 = sample(self.attack,1)
                tempList3 = sample(self.explore,1)
                self.actions.append(tempList[0])
                self.actions.append(tempList2[0])
                self.actions.append(tempList3[0])
        else:
            # . set exploring mode
            for i in range(1000):
                tempList = sample(self.explore,1)
                self.actions.append(tempList[0])

# +-+-+-+-+-+-+-+-+-+-+ +-+-+-+-+-+
# |D|E|P|L|O|Y|M|E|N|T| |L|O|G|I|C|
# +-+-+-+-+-+-+-+-+-+-+ +-+-+-+-+-+

# - Setup logger

botLogger = logging.getLogger(__name__)
f_handler = logging.FileHandler('vizdoom_run.log')
f_handler.setLevel(logging.ERROR)
f_format = logging.Formatter('%(asctime)s - %(process)d - %(levelname)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
f_handler.setFormatter(f_format)
botLogger.addHandler(f_handler)

# - Main banner printing

