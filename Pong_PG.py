# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 14:23:21 2018

@author: Abhimanyu
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 16:40:14 2018

@author: Abhimanyu
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 15:01:41 2017

@author: Abhimanyu
"""

""" Trains an agent with (stochastic) Policy Gradients on Pong. Uses OpenAI Gym. """
import numpy as np
import pickle
import gym
import tensorflow as tf
import itertools
import matplotlib 
import os
os.chdir(r'C:\Users\Abhimanyu\Documents\Reinforcement_Learning\Progs')
import plotting
import collections


matplotlib.style.use('ggplot')
# hyperparameters
H = 200 # number of hidden layer neurons
batch_size = 10 # every how many episodes to do a param update?
learning_rate = 1e-3
gamma = 0.99 # discount factor for reward
decay_rate = 0.99 # decay factor for RMSProp leaky sum of grad^2
resume = False # resume from previous checkpoint?
render = True

# model initialization
D = 80 * 80 # input dimensionality: 80x80 grid

def sigmoid(x): 
  return 1.0 / (1.0 + np.exp(-x)) # sigmoid "squashing" function to interval [0,1]

def prepro(I):
  """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
  I = I[35:195] # crop
  I = I[::2,::2,0] # downsample by factor of 2
  I[I == 144] = 0 # erase background (background type 1)
  I[I == 109] = 0 # erase background (background type 2)
  I[I != 0] = 1 # everything else (paddles, ball) just set to 1
  return I.astype(np.float).ravel()

def discount_rewards(r):
  """ take 1D float array of rewards and compute discounted reward """
  discounted_r = np.zeros_like(r)
  running_add = 0
  for t in reversed(range(0, r.size)):
    if r[t] != 0: running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
    running_add = running_add * gamma + r[t]
    discounted_r[t] = running_add
  return discounted_r

class PolicyEstimator():
    """
    Policy Function approximator. 
    """
    
    def __init__(self, learning_rate=0.0003, scope="policy_estimator"):
        with tf.variable_scope(scope):
            self.state = tf.placeholder(tf.float32, [None,6400], "state")
            self.action = tf.placeholder(dtype=tf.int32,shape=[None], name="action")
            self.target = tf.placeholder(dtype=tf.float32, name="target")

            # This is just table lookup estimator

            self.hidden = tf.contrib.layers.fully_connected(
                inputs=self.state,
                num_outputs=200,
                activation_fn=tf.nn.relu)
            self.output_layer=tf.contrib.layers.fully_connected(
                inputs=self.hidden,
                num_outputs=2,
                activation_fn=tf.nn.softmax)
            self.oneHotAction=tf.one_hot(self.action,2)
            self.action_probs=self.output_layer
            self.action_probs2 = tf.reduce_sum(self.action_probs*self.oneHotAction,1)
            #self.picked_action_prob = tf.gather(self.action_probs, self.action)

            # Loss and train op
            self.loss = tf.reduce_mean(-tf.log(self.action_probs2) * self.target)

            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.train_op = self.optimizer.minimize(
                self.loss, global_step=tf.contrib.framework.get_global_step())
    
    def predict(self, state, sess=None):
        sess = sess or tf.get_default_session()
        return sess.run(self.action_probs, { self.state: np.expand_dims(state,0) })

    def update(self, state, target, action, sess=None):
        sess = sess or tf.get_default_session()
        feed_dict = { self.state: state, self.target: target, self.action: action  }
        _, loss = sess.run([self.train_op, self.loss], feed_dict)
        return loss

env = gym.make("Pong-v0")

#import matplotlib
#matplotlib.pyplot.imshow(observation)
#observation, reward, done, info= env.step(1)
#matplotlib.pyplot.imshow(observation)
#observation, reward, done, info= env.step(2)
#matplotlib.pyplot.imshow(observation)
#observation, reward, done, info= env.step(3)
#

def policy_gradient(env, estimator_policy, num_episodes, discount_factor=1.0):
    """
    Actor Critic Algorithm. Optimizes the policy 
    function approximator using policy gradient.
    
    Args:
        env: OpenAI environment.
        estimator_policy: Policy Function to be optimized 
        estimator_value: Value function approximator, used as a baseline
        num_episodes: Number of episodes to run for
        discount_factor: Time-discount factor
    
    Returns:
        An EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """
    running_reward=None
    # Keeps track of useful statistics
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))    
    
    Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])
    
    for i_episode in range(num_episodes):
        # Reset the environment and pick the fisrst action
        state = env.reset()
        
        episode = []
        reward_ep=[]
        ap=[]
        s=[]
        a=[]
        prev_x=np.zeros(D)
        # One step in the environment
        prev_x = None
        cur_x = prepro(state)
        for t in itertools.count():
            if render: env.render()
            
            x = cur_x - prev_x if prev_x is not None else np.zeros(D)
            prev_x = cur_x
            # Take a step
            action_probs = estimator_policy.predict(x)
            #print(action_probs,"knsnka")
            action_probs=action_probs[0]
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            a.append(action)
            next_state, reward, done, _ = env.step(action+2)
            ap.append(action_probs)
            next_x=(prepro(next_state))
            # Keep track of the transition
            episode.append(Transition(
              state=x, action=action, reward=reward, next_state=(next_x-cur_x), done=done))
            s.append(x)
            reward_ep.append(reward)
            # Update statistics
            stats.episode_rewards[i_episode] += (reward)
            stats.episode_lengths[i_episode] = t
            
           

            if done:
                break
                
            cur_x = next_x
        print("\rStep {} @ Episode {}/{} ({})".format(
                    t, i_episode + 1, num_episodes, stats.episode_rewards[i_episode]), end="")
        discounted_epr = discount_rewards(np.array(reward_ep,dtype=float))
        discounted_epr-=np.mean(discounted_epr)
        discounted_epr/=np.std(discounted_epr)
        #print(discounted_epr)
        running_reward = stats.episode_rewards[i_episode] if running_reward is None else running_reward * 0.93 + stats.episode_rewards[i_episode] * 0.07
        print(running_reward)
        s=np.vstack(s)
        a=np.vstack(a)
        a=np.squeeze(a)
        #print(s.shape)
        #episode=np.array(episode)
#        for i,k in enumerate(episode):
##                matplotlib.pyplot.imshow(k[0].reshape(80,80))
##                matplotlib.pyplot.show()
#                #print(ap[i])
#                estimator_policy.update(k[0],discounted_epr[i],k[1])
        estimator_policy.update(s,discounted_epr,a)
        
    return stats

tf.reset_default_graph()

global_step = tf.Variable(0, name="global_step", trainable=False)
policy_estimator = PolicyEstimator()
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    # Note, due to randomness in the policy the number of episodes you need to learn a good
    # policy may vary. ~300 seemed to work well for me.
    stats = policy_gradient(env, policy_estimator, 1500)
    save_path = saver.save(sess, r"C:\Users\Abhimanyu\Documents\Reinforcement_Learning\Weights\model_new.ckpt")

#with tf.Session() as sess:
#    saver.restore(sess,r"C:\Users\Abhimanyu\Documents\Reinforcement_Learning\Weights\model_2.1.ckpt")
#    stats = policy_gradient(env, policy_estimator, 1500)
#    save_path = saver.save(sess, r"C:\Users\Abhimanyu\Documents\Reinforcement_Learning\Weights\model_3.ckpt")
#    
#    
plotting.plot_episode_stats(stats, smoothing_window=10)


