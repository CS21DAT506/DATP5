import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import tensorflow_probability.python as tfp
from agent.actor_critic_network import ActorCriticNetwork
import numpy as np
from agent.agent_base import AgentBase


from settings.SettingsAccess import settings



class Agent(AgentBase):
    def __init__(self, alpha=0.0003, gamma=0.99, n_actions=4):
        max_acceleration = settings.max_acceleration

        self.gamma = gamma
        self.n_actions = n_actions
        self.action = None
        self.actions = [
            [max_acceleration,0],[-max_acceleration,0],
            [0,max_acceleration],[0,-max_acceleration],
            # [max_acceleration/2, max_acceleration/2], [max_acceleration/2, -max_acceleration/2],
            # [-max_acceleration/2, max_acceleration/2], [-max_acceleration/2, -max_acceleration/2],
        ]

        self.actor_critic = ActorCriticNetwork(n_actions=n_actions)
        self.actor_critic.compile(optimizer=Adam(learning_rate=alpha))

    def __call__(self, target_pos):
        self.target_pos = target_pos
        self.state = None
        self.new_state = None
        self.reward = None
        self.done = False

        self.score = 0

        self.time = -1
        self.output = self.actions[0]

        return self


    def _get_agent_gravity(self, agent_pos, sim):
        agent_acc =  np.array( (0, 0) )

        for i in range(1, len(sim.particles)):
            particle = sim.particles[i]
            distance = np.array( (particle.x, particle.y) ) - agent_pos 
            agent_acc = agent_acc + particle.m * distance / np.linalg.norm(distance)**3
        
        return agent_acc * sim.G

    def choose_action(self, observation):
        state = tf.convert_to_tensor([observation])
        self.state = observation
        self.new_state = self.state

        _, probs = self.actor_critic(state)

        action_probabilities = tfp.distributions.Categorical(probs=probs)
        # print(state)
        # print(probs)
        # input("")
        action = action_probabilities.sample()

        self.action = action

        return action.numpy()[0]

    def save_models(self):
        self.actor_critic.save_weights(self.actor_critic.checkpoint_file)

    def load_models(self):
        self.actor_critic.load_weights(self.actor_critic.checkpoint_file)

    def cal_reward(self, sim):
        agent = sim.particles[0]
        agent_pos = np.array((agent.x, agent.y, agent.z))

        dist = np.sqrt((self.target_pos[0] - agent.x)
                       ** 2 + (self.target_pos[1] - agent.y)**2)
        if (dist < 50):
            self.reward = 3000.0
            self.done = True
            print(""*20, "\rSUCCES", end="")
        else:
            if sim.t > 29.99:
                self.done = True
            else:
                self.done = False
            # self.reward = -1.0
            self.reward = -dist

        self.score += self.reward
        return self.reward

    def learn(self):
        state = tf.convert_to_tensor([self.state])
        new_state = tf.convert_to_tensor([self.new_state])
        reward = tf.convert_to_tensor(np.array([self.reward]))

        with tf.GradientTape(persistent=True) as tape:
            state_value, probs = self.actor_critic(state)
            new_state_value, _ = self.actor_critic(new_state)

            state_value = tf.squeeze(state_value)
            new_state_value = tf.squeeze(new_state_value)

            action_probs = tfp.distributions.Categorical(probs=probs)
            
            log_prob = action_probs.log_prob(self.action)
            action_probs = np.array(probs[0])

            delta = reward + self.gamma * new_state_value*(1-int(self.done)) - state_value
            actor_loss = -log_prob * delta
            critic_loss = delta**2

            total_loss = actor_loss + critic_loss

        # with tf.GradientTape(persistent=True) as tape:
        #     state_value, probs = self.actor_critic(state)
        #     new_state_value, _ = self.actor_critic(new_state)

        #     state_value = tf.squeeze(state_value)
        #     new_state_value = tf.squeeze(new_state_value)

        #     action_probs = tfp.distributions.Categorical(probs=probs)

        #     log_prob = action_probs.log_prob(self.action)
        #     # print(log_prob)
        #     # input("")
        #     action_probs = np.array(probs[0])

        #     delta = reward + self.gamma * new_state_value*(1-int(self.done)) - state_value
        #     actor_loss = -log_prob * delta
        #     critic_loss = delta**2

        #     total_loss = actor_loss + critic_loss

        with tf.GradientTape(persistent=True) as tape:
            state_value, probs = self.actor_critic(state)
            new_state_value, _ = self.actor_critic(new_state)

            state_value = tf.squeeze(state_value)
            new_state_value = tf.squeeze(new_state_value)

            action_probs = tfp.distributions.Categorical(probs=probs)

            log_prob = action_probs.log_prob(self.action)
            action_probs = np.array(probs[0])

            delta = reward + self.gamma * \
                new_state_value*(1-int(self.done)) - state_value
            actor_loss = -log_prob * delta
            critic_loss = delta**2

            total_loss = actor_loss + critic_loss

        gradient = tape.gradient(total_loss, self.actor_critic.trainable_variables,
                                unconnected_gradients=tf.UnconnectedGradients.ZERO)
        self.actor_critic.optimizer.apply_gradients(zip(gradient, self.actor_critic.trainable_variables))

    def get_thrust(self, sim):
        if self.done:
            return np.zeros(3)

        agent = sim.particles[0]

        agent_pos = np.array([agent.x, agent.y])
        agent_vel = np.array([agent.vx, agent.vy])
        gravity = self._get_agent_gravity(agent_pos, sim)

        observation = [*self.target_pos, *agent_pos, *agent_vel, *gravity]

        if self.state is None:
            self.state = observation
        
        # print(f"Get thrust. sim.t: {sim.t}")

        if self.time != sim.t:
            self.time = sim.t
            action = self.choose_action(observation)
            self.output = action
            print("time:", self.time, end=" | ")
            print("action:", action)
            self.output = self.actions[self.output]


            self.cal_reward(sim)
            self.learn()

        # print("action_space ", self.action_space)
        # print("action ", self.output)
        return np.array([*self.output, 0])
