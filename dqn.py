import tensorflow as tf
import numpy as np


class ExperienceMemory:
    def __init__(self, mem_size, input_dims, n_actions):
        self.mem_size = mem_size
        self.mem_counter = 0
        self.n_actions = n_actions
        self.state_memory = np.zeros((self.mem_size, *input_dims))
        self.action_memory = np.zeros((self.mem_size, n_actions),
                                      dtype=np.int8)
        self.reward_memory = np.zeros(self.mem_size)
        self.new_state_memory = np.zeros((self.mem_size, *input_dims))
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.int8)

    def store(self, state, action, reward, state_, terminal):
        index = self.mem_counter % self.mem_size
        self.state_memory[index] = state
        actions = np.zeros(self.n_actions)
        actions[action] = 1.0
        self.action_memory[index] = actions
        self.reward_memory[index] = reward
        self.new_state_memory[index] = state_
        # 0 for terminated, easy for target_q calculation
        self.terminal_memory[index] = 1 - int(terminal)
        self.mem_counter += 1

    def sample(self, batch_size):
        sample_nums = np.random.choice(min(self.mem_size, self.mem_counter), batch_size)

        states = self.state_memory[sample_nums]
        actions = self.action_memory[sample_nums]
        rewards = self.reward_memory[sample_nums]
        new_states = self.new_state_memory[sample_nums]
        terminals = self.terminal_memory[sample_nums]

        return states, actions, rewards, new_states, terminals


class DQN:
    def __init__(self, lr, n_actions, name, input_dims, fc1_units, fc2_units):
        self.lr = lr
        self.n_actions = n_actions
        self.name = name
        self.fc1_units = fc1_units
        self.fc2_units = fc2_units
        self.sess = tf.Session()
        with tf.variable_scope(self.name):
            self.input = tf.placeholder(tf.float32,
                                        shape=[None, *input_dims],
                                        name='inputs')
            self.actions = tf.placeholder(tf.float32,
                                          shape=[None, self.n_actions],
                                          name='action_taken')
            self.q_values = self.build_network()
            self.q_target = tf.placeholder(tf.float32,
                                           shape=[None, self.n_actions],
                                           name='q_target')
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_values, self.q_target))
            self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
        self.sess.run(tf.global_variables_initializer())

    def build_network(self):
        flat = tf.layers.flatten(self.input)
        dense1 = tf.layers.dense(flat, units=self.fc1_units,
                                 activation=tf.nn.relu)
        dense2 = tf.layers.dense(dense1, units=self.fc1_units,
                                 activation=tf.nn.relu)
        return tf.layers.dense(dense2, units=self.n_actions)


class OriginalDQNAgent:
    """This is the original DQN proposed in 2013
    Implemented based on https://arxiv.org/abs/1312.5602
    """
    def __init__(self, lr, gamma, mem_size, n_actions, batch_size,
                 input_dims=(210, 160, 4), epsilon_start=0.1,
                 epsilon_dec=True, epsilon_end=0.01, fc1_units=32, fc2_units=64):
        self.action_space = [i for i in range(n_actions)]
        self.n_actions = n_actions
        self.gamma = gamma
        self.mem_size = mem_size
        self.epsilon = epsilon_start
        self.epsilon_dec = epsilon_dec
        self.epsilon_min = epsilon_end
        self.batch_size = batch_size
        self.q_net = DQN(lr, n_actions, input_dims=input_dims, name='q_net',
                         fc1_units=fc1_units, fc2_units=fc2_units)
        self.experience_memory = ExperienceMemory(self.mem_size, input_dims, self.n_actions)

    def record(self, state, action, reward, new_state, terminal):
        self.experience_memory.store(state, action, reward, new_state, terminal)

    def act(self, state, test=False):
        state = state[np.newaxis, :]
        rand = np.random.random()
        if rand < self.epsilon and not test:
            action = np.random.choice(self.action_space)
        else:
            actions = self.q_net.sess.run(self.q_net.q_values,
                                          feed_dict={self.q_net.input: state})
            action = np.argmax(actions)

        return action

    def learn(self):
        if self.experience_memory.mem_counter < self.batch_size:
            return

        states, actions, rewards, new_states, terminals = self.experience_memory.sample(self.batch_size)

        q_target = self.q_net.sess.run(self.q_net.q_values,
                                       feed_dict={self.q_net.input: states})
        q_next = self.q_net.sess.run(self.q_net.q_values,
                                     feed_dict={self.q_net.input: new_states})

        # update only the corresponding action
        action_indices = np.dot(actions, np.array(self.action_space, dtype=np.int8))
        batch_indices = np.arange(self.batch_size, dtype=np.int32)

        q_target[batch_indices, action_indices] = rewards + self.gamma * np.max(q_next, axis=1) * terminals

        self.q_net.sess.run(self.q_net.optimizer,
                            feed_dict={self.q_net.input: states,
                                       self.q_net.actions: actions,
                                       self.q_net.q_target: q_target})

        if self.epsilon_dec and self.epsilon * 0.99 >= self.epsilon_min:
            self.epsilon *= 0.99


class NatureDQNAgent:
    """This is the DQN published on Nature in 2015
    Implemented based on https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf
    """
    pass
