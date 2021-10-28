from math import sqrt
import numpy as np
import tensorflow as tf
from actor_critic_network import ActorCriticNetwork
import tensorflow_probability.python as tfp
from tensorflow.keras.optimizers import Adam

from settings.SettingsAccess import settings


class ActorCriticAgent:
    def __init__(self, alpha=0.0003, gamma=0.99):#, n_actions=2):
        max_acceleration = settings.max_acceleration
        actions = [
            [max_acceleration,0],[-max_acceleration,0],
            [0,max_acceleration],[0,-max_acceleration],
            [max_acceleration/2, max_acceleration/2], [max_acceleration/2, -max_acceleration/2],
            [-max_acceleration/2, max_acceleration/2], [-max_acceleration/2, -max_acceleration/2],
        ]

        n_actions = len(actions)

        self.gamma = gamma
        self.n_actions = n_actions
        self.action = None
        self.action_space = [i for i in range(self.n_actions)]

        self.actor_critic = ActorCriticNetwork(n_actions=n_actions)

        self.actor_critic.compile(optimizer=Adam(learning_rate=alpha))

    def call(self, target_point):
        self.target_point = target_point
        self.state = None
        self.new_state = None
        self.reward = None

    def choose_action(self, observation):
        self.state = observation
        self.new_state = self.state

        state = tf.convert_to_tensor([observation])
        _, probs = self.actor_critic(state)

        action_probabilities = tfp.distributions.Categorical(probs=probs)
        action = action_probabilities.sample()
        self.action = action

        return action.numpy()[0]

    def save_models(self):
        self.actor_critic.save_weights(self.actor_critic.checkpoint_file)

    def load_models(self):
        self.actor_critic.load_weights(self.actor_critic.checkpoint_file)

    def cal_reward(self, sim):
        agent = sim.particles[0]
        agent_pos = np.array( (agent.x, agent.y, agent.z) )

        dist = np.sqrt((self.target_point[0] - agent.x)**2 + (self.target_point[1] - agent.y)**2) 
        if (dist < 5):
            self.reward = 5
        else:
            self.reward = -1
        return self.reward

    def learn(self, done):
        state = tf.convert_to_tensor([self.state])
        new_state = tf.convert_to_tensor([self.new_state])
        reward = tf.convert_to_tensor(self.reward)

        with tf.GradientTape(persistent=True) as tape:
            state_value, probs = self.actor_critic(state)
            new_state_value, _ = self.actor_critic(new_state)

            state_value = tf.squeeze(state_value)
            new_state_value = tf.squeeze(new_state_value)

            action_probs = tfp.distributions.Categorical(probs=probs)
            log_prob = action_probs.log_prob(self.action)

            delta = reward + self.gamma * \
                new_state_value*(1-int(done)) - state_value
            actor_loss = -log_prob * delta
            critic_loss = delta**2

            total_loss = actor_loss + critic_loss

        gradient = tape.gradient(
            total_loss, self.actor_critic.trainable_variables)
        self.actor_critic.optimizer.apply_gradients(
            zip(gradient, self.actor_critic.trainable_variables))

    def _get_agent_acceleration(self, sim):
        nn_input_data = [self.target_pos[0], self.target_pos[1]]

        for particle in sim.particles:
            nn_input_data.extend([particle.x, particle.y, particle.vx, particle.vy, particle.m])

        # start_t = time.time()
        res = np.append( self.choose_action([nn_input_data])[0], [0] ) # add 0 as the z-axis
        # print(f"Finished predicting. Time spent: {time.time() - start_t}")
        return res

    def get_thrust(self, archive):
        sim = archive
        # print(f"Get thrust. sim.t: {sim.t}")

        if self.time != sim.t:
            self.time = sim.t
            self.output = self._get_agent_acceleration(sim)

        self.cal_reward(sim)
        self.learn()

        return self.output

        
