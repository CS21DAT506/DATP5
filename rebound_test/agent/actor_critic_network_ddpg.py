import os
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense
import numpy as np
import tensorflow as tf


class ActorCriticNetworkDDPG(keras.Model):
    def __init__(self, n_actions, name="actor_critic", chkpt_dir='tmp/actor_critic'):
        super(ActorCriticNetworkDDPG, self).__init__()
        self.an1_dims = 256
        self.an2_dims = 256
        self.cn1_dims = 256
        self.cn2_dims = 256
        self.n_actions = n_actions
        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.an_checkpoint_file = os.path.join(self.checkpoint_dir, name + "_ac_an")
        self.cn_checkpoint_file = os.path.join(self.checkpoint_dir, name + "_ac_cn")

        self.an1 = Dense(self.an1_dims, activation="relu")
        self.an2 = Dense(self.an2_dims, activation="relu")
        self.cn1 = Dense(self.cn1_dims, activation="relu")
        self.cn2 = Dense(self.cn2_dims, activation="relu")

        self.q = Dense(1, activation=None)
        self.pi = Dense(n_actions, activation="tanh")
        # self.pi = Dense(n_actions, activation=None)

        self.target_actor = keras.models.Sequential()
        self.target_actor.add(self.an1)
        self.target_actor.add(self.an2)
        self.target_actor.add(self.pi)

        self.target_critic = keras.models.Sequential()
        self.target_critic.add(self.cn1)
        self.target_critic.add(self.cn2)
        self.target_critic.add(self.q)
        
        self.actor = keras.models.Sequential()
        self.actor.add(self.an1)
        self.actor.add(self.an2)
        self.actor.add(self.pi)

        self.critic = keras.models.Sequential()
        self.critic.add(self.cn1)
        self.critic.add(self.cn2)
        self.critic.add(self.q)

    def call(self, state):
        """
        returns q, pi
        """
        # an_value = self.an1(state)
        # an_value = self.an2(an_value)

        # cn_value = self.cn1(state)
        # cn_value = self.cn2(cn_value)

        # pi = self.pi(an_value)
        # v = self.v(cn_value)

        pi = self.actor(state)
        
        q_input = tf.concat([state, pi], axis=1)
        # q_input = np.array([[*state.numpy()[0], *pi.numpy()[0]]])
        q = self.critic(q_input)

        return q, pi
    
    def call_target(self, state):
        target_pi = self.target_actor(state)
        
        target_q_input = tf.concat([state, target_pi], axis=1)
        # target_q_input = np.array([[*state.numpy()[0], *target_pi.numpy()[0]]])
        target_q = self.target_critic(target_q_input)

        return target_q, target_pi
