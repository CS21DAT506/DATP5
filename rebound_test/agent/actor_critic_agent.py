from agent.agent_base import AgentBase
import numpy as np
import tensorflow as tf
from agent.actor_critic_network import ActorCriticNetwork
from random import random
import tensorflow_probability.python as tfp
from tensorflow.keras.optimizers import Adam

from settings.SettingsAccess import settings

class ActorCriticAgent(AgentBase):
    def __init__(self, alpha=0.0003, gamma=0.99):#, n_actions=2):
        max_acceleration = settings.max_acceleration
        self.actions = [
            [max_acceleration,0],[-max_acceleration,0],
            [0,max_acceleration],[0,-max_acceleration],
            [max_acceleration/2, max_acceleration/2], [max_acceleration/2, -max_acceleration/2],
            [-max_acceleration/2, max_acceleration/2], [-max_acceleration/2, -max_acceleration/2],
        ]

        self.gamma = gamma
        self.n_actions = len(self.actions)
        self.action = None

        self.actor_critic = ActorCriticNetwork(n_actions=self.n_actions)
        self.actor_critic.compile(optimizer=Adam(learning_rate=alpha))

    def __call__(self, target_pos):
        self.target_pos = target_pos
        self.state = None
        self.new_state = None
        self.reward = None
        self.done = False

        self.score = 0

        self.time = -1
        self.output = np.array([])

        return self


    def _get_agent_gravity(self, agent_pos, sim):
        agent_acc =  np.array( (0, 0, 0) )

        for i in range(1, len(sim.particles)):
            particle = sim.particles[i]
            distance = np.array( (particle.x, particle.y, particle.z) ) - agent_pos 
            agent_acc = agent_acc + particle.m * distance / np.linalg.norm(distance)**3
        
        return agent_acc * sim.G

    def choose_action(self, observation):
        self.state = observation
        self.new_state = self.state

        state = tf.convert_to_tensor([observation])
        _, probs = self.actor_critic(state)

        action_probabilities = tfp.distributions.Categorical(probs=probs)
        # print("action_probabilities:", action_probabilities)

        action = action_probabilities.sample()
        # print("probs:", probs)

        self.action = action
        # print("action:", action[0])
        # input("")

        return action[0]

        # state = tf.convert_to_tensor([observation])
        # _, probs = self.actor_critic(state)

        # action_probabilities = tfp.distributions.Categorical(probs=probs)
        # action_probs = np.array(probs[0])

        # print(action_probabilities)

        # # r_val = random()
        # # sum_a = 0
        # # # print(action_probs)

        # # for i in range(len(action_probs)):
        # #     sum_a += action_probs[i]

        # #     if sum_a > r_val:
        # #         self.action = i
        # #         break
        # self.action = action_probabilities.sample()
        # print(self.action)
        
        # # print("i: ", self.action)
        # # print("array: ", np.array(self.action))
        # # input("")


        # # return np.array(self.action)
        # return self.action

    def save_models(self):
        self.actor_critic.save_weights(self.actor_critic.checkpoint_file)

    def load_models(self):
        self.actor_critic.load_weights(self.actor_critic.checkpoint_file)

    def cal_reward(self, sim):
        agent = sim.particles[0]
        agent_pos = np.array( (agent.x, agent.y, agent.z) )

        dist = np.sqrt((self.target_pos[0] - agent.x)**2 + (self.target_pos[1] - agent.y)**2) 
        if (dist < 50):
            self.reward = 3000.0
            self.done = True
            print(""*20, "\rSUCCES", end="")
        else:
            if sim.t >29.99:
                self.done = True
            else:
                self.done = False
            self.reward = -1.0
            # self.reward = -dist


        self.score += self.reward
        return self.reward

    def learn(self, reward=None):
        if reward is None:
            reward = self.reward
            self.score += reward

        state = tf.convert_to_tensor([self.state])
        new_state = tf.convert_to_tensor([self.new_state])
        reward = tf.convert_to_tensor(self.reward)

        with tf.GradientTape() as tape:
            state_value, probs = self.actor_critic(state)
            new_state_value, _ = self.actor_critic(new_state)
            state_value = tf.squeeze(state_value)
            new_state_value = tf.squeeze(new_state_value)

            action_probs = tfp.distributions.Categorical(probs=probs)
            log_prob = action_probs.log_prob(self.action)

            # print("reward:", reward)
            # print("gamma:", self.gamma)
            # print("new_state_value:", new_state_value)
            # print("done:", self.done)
            # print("state_value:", state_value)
            # input("")

            delta = reward + self.gamma*new_state_value * (1-int(self.done)) - state_value
            actor_loss = -log_prob*delta
            critic_loss = delta**2
            total_loss = actor_loss + critic_loss


        # with tf.GradientTape(persistent=True) as tape:
        #     state_value, probs = self.actor_critic(state)
        #     new_state_value, _ = self.actor_critic(new_state)

        #     state_value = tf.squeeze(state_value)
        #     new_state_value = tf.squeeze(new_state_value)

        #     state = np.array([self.state])
        #     new_state = np.array([self.new_state])
        #     reward = np.array(self.reward)

        #     # action_probs = tfp.distributions.Categorical(probs=probs)
        #     # log_prob = action_probs.log_prob(self.action)
        #     action_probs = np.array(probs[0])
            
        #     log_prob = np.log(action_probs[np.argmax(np.array(self.action))])
        #     log_prob = tf.convert_to_tensor(np.array(log_prob))

        #     # print("reward", reward)
        #     # print("new_state_value*(1-int(self.done))", new_state_value*(1-int(self.done)))
        #     # print("state_value", state_value)
        #     # input()
        #     delta = reward #+ self.gamma * new_state_value*(1-int(self.done)) - state_value
        #     actor_loss = -log_prob * delta
        #     critic_loss = delta**2

        #     total_loss = actor_loss + critic_loss

        # print("\n\nLOSS:", total_loss)
        gradient = tape.gradient(
            total_loss, self.actor_critic.trainable_variables,
            )

        self.actor_critic.optimizer.apply_gradients(zip(gradient, self.actor_critic.trainable_variables))

    def _get_agent_gravity(self, agent_pos, sim):
        agent_acc =  np.array( [0, 0] )

        for i in range(1, len(sim.particles)):
            particle = sim.particles[i]
            distance = np.array( (particle.x, particle.y) ) - agent_pos 
            agent_acc = agent_acc + particle.m * distance / np.linalg.norm(distance)**3
        
        return agent_acc * sim.G

    def get_thrust(self, sim):
        if self.done:
            return np.zeros(3)

        agent = sim.particles[0]

        agent_pos = np.array([agent.x, agent.y])
        agent_vel = np.array([agent.vx, agent.vy])
        gravity = self._get_agent_gravity(agent_pos, sim)

        observation = [*self.target_pos, *agent_pos, *agent_vel, *gravity]
        
        # print(f"Get thrust. sim.t: {sim.t}")

        if self.time != sim.t:
            self.time = sim.t
            # print("time:", self.time)
            action = self.choose_action(observation)
            if action >= self.n_actions:
                pass
                # print("self.output:", self.output)
                # input("")
            else:
                self.output = action
                self.output = self.actions[self.output]


            reward = self.cal_reward(sim)
            self.learn(reward=reward)

        # print("action_space ", self.action_space)
        # print("action ", self.output)
        return np.array([*self.output, 0])
