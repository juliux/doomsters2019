#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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

# Q-learning settings
learning_rate = 0.00025
# learning_rate = 0.0001
discount_factor = 0.99
epochs = 20 # JOPI MOD 20
learning_steps_per_epoch = 2000 # JOPI MOD 2000
replay_memory_size = 10000

# NN learning settings
batch_size = 64

# Training regime
test_episodes_per_epoch = 100

# Other parameters
frame_repeat = 12
resolution = (30, 45)
episodes_to_watch = 10

# Play with this many bots
bots = 0

# TODO move to argparser
save_model = False
load_model = True
skip_learning = True

# Configuration file path
DEFAULT_MODEL_SAVEFILE = "/home/doomsters/model0/"
#DEFAULT_CONFIG = "../../scenarios/simpler_basic.cfg"
# JOPI MOD
DEFAULT_CONFIG = "../../scenarios/vdka.cfg"
# //JOPI MOD

# config_file_path = "../../scenarios/rocket_basic.cfg"
# config_file_path = "../../scenarios/basic.cfg"


# Converts and down-samples the input image
#def preprocess(img):
#    img = skimage.transform.resize(img, resolution)
#    img = img.astype(np.float32)
#    return img

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


# Creates and initializes ViZDoom environment.
def initialize_vizdoom(config_file_path):
    print("Initializing doom...")
    game = vzd.DoomGame()
    game.load_config(config_file_path)
    game.set_window_visible(False)
    #game.set_mode(vzd.Mode.PLAYER)
    game.set_mode(vzd.Mode.ASYNC_PLAYER)
    game.set_screen_format(vzd.ScreenFormat.GRAY8)
    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
    #game.add_game_args("-join 10.44.244.114")
    #game.add_game_args("+name DOOMSTERS +colorset 0")

    # JOPI MOD
    #game.set_living_reward(-10)
    #game.set_death_penalty(200)
    #game.add_game_args("-host 1 -deathmatch +timelimit 10.0 "
     #      "+sv_forcerespawn 1 +sv_noautoaim 1 +sv_respawnprotect 1 +sv_spawnfarthest 1 +sv_nocrouch 1 "
     #       "+viz_respawn_delay 10 +viz_nocheat 1")
    # Sets map to start (scenario .wad files can contain many maps).
    game.set_doom_map("map01")
    # //JOPI MOD

    #game.init()
    #game.send_game_command("removebots")
    #for i in range(bots):
    #    game.send_game_command("addbot")

    print("Doom initialized.")
    #player_number = int(game.get_game_variable(GameVariable.PLAYER_NUMBER))
    #last_frags = 0
    return game

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

if __name__ == '__main__':
    parser = ArgumentParser("ViZDoom example showing how to train a simple agent using simplified DQN.")
    parser.add_argument(dest="config",
                        default=DEFAULT_CONFIG,
                        nargs="?",
                        help="Path to the configuration file of the scenario."
                             " Please see "
                             "../../scenarios/*cfg for more scenarios.")

    args = parser.parse_args()

    # Create Doom instance
    game = initialize_vizdoom(args.config)

    # Action = which buttons are pressed
    #n = game.get_available_buttons_size()
    #actions = [list(a) for a in it.product([0, 1], repeat=n)]

    myPlayer = playerBehaviour()
    myPlayer.playerBehaviour(3)
    actions = myPlayer.actions

    # Create replay memory which will store the transitions
    memory = ReplayMemory(capacity=replay_memory_size)
    # . GUP detail

    session = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    
    # Runs the op.
    #print(sess.run(c))

    #session = tf.Session()
    learn, get_q_values, get_best_action = create_network(session, len(actions))
    saver = tf.train.Saver()
    if load_model:
        print("Loading model from: ", DEFAULT_MODEL_SAVEFILE)
        saver.restore(session, DEFAULT_MODEL_SAVEFILE)
    else:
        init = tf.global_variables_initializer()
        session.run(init)
    print("Starting the training!")

    time_start = time()
    if not skip_learning:
        for epoch in range(epochs):
            print("\nEpoch %d\n-------" % (epoch + 1))
            train_episodes_finished = 0
            train_scores = []

            print("Training...")
            game.new_episode()
            for learning_step in trange(learning_steps_per_epoch, leave=False):
                perform_learning_step(epoch)
                if game.is_episode_finished():
                    score = game.get_total_reward()
                    train_scores.append(score)
                    game.new_episode()
                    train_episodes_finished += 1

            print("%d training episodes played." % train_episodes_finished)

            train_scores = np.array(train_scores)

            print("Results: mean: %.1f±%.1f," % (train_scores.mean(), train_scores.std()), \
                  "min: %.1f," % train_scores.min(), "max: %.1f," % train_scores.max())

            print("\nTesting...")
            test_episode = []
            test_scores = []
            for test_episode in trange(test_episodes_per_epoch, leave=False):
                game.new_episode()
                while not game.is_episode_finished():
                    state = preprocess(game.get_state().screen_buffer)
                    best_action_index = get_best_action(state)

                    game.make_action(actions[best_action_index], frame_repeat)
                r = game.get_total_reward()
                test_scores.append(r)

            test_scores = np.array(test_scores)
            print("Results: mean: %.1f±%.1f," % (
                test_scores.mean(), test_scores.std()), "min: %.1f" % test_scores.min(),
                  "max: %.1f" % test_scores.max())

            print("Saving the network weigths to:", DEFAULT_MODEL_SAVEFILE)
            saver.save(session, DEFAULT_MODEL_SAVEFILE)

            print("Total elapsed time: %.2f minutes" % ((time() - time_start) / 60.0))

    game.close()
    print("======================================")
    print("Training finished. It's time to watch!")

    # Reinitialize the game with window visible
    game.set_window_visible(True)
    game.set_mode(vzd.Mode.ASYNC_PLAYER)
    game.add_game_args("-join 10.44.244.114")
    game.add_game_args("+name DOOMSTERS +colorset 0")

    game.init()

    
    #game.send_game_command("removebots")
    #for i in range(bots):
    #    game.send_game_command("addbot")

    #for _ in range(episodes_to_watch):
    #    game.new_episode()
    while not game.is_episode_finished():
        state = preprocess(game.get_state().screen_buffer)
        best_action_index = get_best_action(state)

        # Instead of make_action(a, frame_repeat) in order to make the animation smooth
        #game.set_action(actions[best_action_index])
        #for _ in range(frame_repeat):
        #game.advance_action()


        #game.make_action(choice(actions))
        game.make_action(actions[best_action_index])



        # Sleep between episodes
    #    sleep(1.0)
    #    score = game.get_total_reward()
    #    print("Total score: ", score)
        if game.is_player_dead():
            #print("Player " + str(player_number) + " died.")
            game.respawn_player()

game.close()
