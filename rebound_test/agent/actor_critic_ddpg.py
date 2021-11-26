import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.python.eager.backprop import GradientTape
import tensorflow_probability.python as tfp
from agent.actor_critic_network_ddpg import ActorCriticNetworkDDPG
import numpy as np
from agent.agent_base import AgentBase

# from settings.SettingsAccess import settings


class AgentDDPG(AgentBase):
    def __init__(self, alpha=0.003, gamma=0.99):
        # max_acceleration = settings.max_acceleration

        self.gamma = gamma
        self.action = None
    
        self.n_actions = 2

        self.actor_critic = ActorCriticNetworkDDPG(n_actions=self.n_actions)

        self.actor_critic.actor.compile(optimizer=Adam(learning_rate=alpha))
        self.actor_critic.critic.compile(optimizer=Adam(learning_rate=alpha))

    def __call__(self, target_pos):
        self.target_pos = target_pos
        self.new_state = None
        self.new_state = None
        self.reward = None
        self.done = False

        self.score = 0

        self.time = -1
        self.output = None

        self.losses = {
            "actor": [],
            "critic": []
        }

        return self


    def _get_agent_gravity(self, agent_pos, sim):
        agent_acc =  np.array((0, 0))

        for i in range(1, len(sim.particles)):
            particle = sim.particles[i]
            distance = np.array((particle.x, particle.y)) - agent_pos 
            agent_acc = agent_acc + particle.m * distance / np.linalg.norm(distance)**3
        
        return agent_acc * sim.G

    def choose_action(self, observation):
        state = np.array([observation])
        q, action = self.actor_critic(state)

        self.action = action

        return q[0], action[0]

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
            self.reward = -1.0
            # self.reward = -dist

        self.score += self.reward
        return self.reward

    def learn(self, state, new_state, reward, done):
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        new_state = tf.convert_to_tensor([new_state], dtype=tf.float32)
        reward = tf.convert_to_tensor(reward, dtype=tf.float32)
        done = tf.convert_to_tensor(float(done), dtype=tf.float32)

        with tf.GradientTape() as critic_tape:
            q, _ = self.actor_critic(state)
            new_q, _ = self.actor_critic.call_target(new_state)


            q_target = reward + tf.convert_to_tensor(self.gamma) * (1-done) * new_q

            critic_loss = (q - q_target)**2
            self.losses["critic"].append(critic_loss)

            critic_gradient = critic_tape.gradient(critic_loss, self.actor_critic.critic.trainable_variables)
        
        self.actor_critic.critic.optimizer.apply_gradients( zip(critic_gradient,
                                                                self.actor_critic.critic.trainable_variables
                                                            )
                                                        )
        with GradientTape() as actor_tape:
            action = self.actor_critic.actor(state)

            actor_loss = -self.actor_critic.critic(tf.concat([state, action], axis=1)) #* action[0][1] 
            self.losses["actor"].append(actor_loss)

            actor_gradient = actor_tape.gradient(actor_loss, self.actor_critic.actor.trainable_variables)
        
        self.actor_critic.actor.optimizer.apply_gradients( zip(actor_gradient,
                                                                self.actor_critic.actor.trainable_variables
                                                            )
                                                        )

        return


    def get_thrust(self, sim):

        if self.time != sim.t:
            self.time = sim.t

            agent = sim.particles[0]

            agent_pos = np.array([agent.x, agent.y])
            agent_vel = np.array([agent.vx, agent.vy])
            gravity = self._get_agent_gravity(agent_pos, sim)

            observation = [*self.target_pos, *agent_pos, *agent_vel, *gravity]

            q, action = self.choose_action(observation)
            self.output = action

            self.state = self.new_state
            self.new_state = observation

            self.reward = self.cal_reward(sim)
            self.score += self.reward

            if not self.state is None:
                self.learn(self.state, self.new_state, self.reward, self.done)

            print("time:", self.time, end=" | ")
            print("reward:", self.reward, end =" | ")
            print("q-val:", q.numpy()[0], end =" | ")
            print("action:", action.numpy())


        
        # print(f"Get thrust. sim.t: {sim.t}")

        # print("action_space ", self.action_space)
        # print("action ", self.output)
    
        if self.done or self.action is None:
            return np.zeros(3)
    
        return np.array([*self.output, 0])
