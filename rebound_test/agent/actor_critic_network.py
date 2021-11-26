import os
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense
import numpy as np


class ActorCriticNetwork(keras.Model):
    def __init__(self, n_actions, fc1_dims=1024, fc2_dims=512, fc3_dims=256, fc4_dims=128, name="actor_critic", chkpt_dir='tmp/actor_critic'):
        super(ActorCriticNetwork, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.fc3_dims = fc3_dims
        self.fc4_dims = fc4_dims
        self.n_actions = n_actions
        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + "_ac")

        self.fc1 = Dense(self.fc1_dims, activation="relu")
        self.fc2 = Dense(self.fc2_dims, activation="relu")
        self.fc3 = Dense(self.fc3_dims, activation="relu")
        self.fc4 = Dense(self.fc4_dims, activation="relu")
        self.v = Dense(1, activation=None)
        self.pi = Dense(n_actions, activation="softmax")

    def call(self, state):
        value = self.fc1(state)
        value = self.fc2(value)
        value = self.fc3(value)
        value = self.fc4(value)

        v = self.v(value)
        pi = self.pi(value)

        return v, pi
