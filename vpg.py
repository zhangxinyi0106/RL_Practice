import tensorflow as tf
import numpy as np

from dqn import DQN, ExperienceMemory


class TrajectoryStore(ExperienceMemory):
    def clear(self):
        self.state_memory = np.zeros((self.mem_size, *self.input_dims))
        self.action_memory = np.zeros((self.mem_size, self.n_actions),
                                      dtype=np.int8)
        self.reward_memory = np.zeros(self.mem_size)
        self.new_state_memory = np.zeros((self.mem_size, *self.input_dims))
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.int8)
        self.mem_counter = 0

    def get_trajectory(self):
        sample_nums = np.arange(self.mem_counter, dtype=np.int32)

        states = self.state_memory[sample_nums]
        actions = self.action_memory[sample_nums]
        rewards = self.reward_memory[sample_nums]
        new_states = self.new_state_memory[sample_nums]
        terminals = self.terminal_memory[sample_nums]

        return states, actions, rewards, new_states, terminals


class PolicyNetwork:
    def __init__(self, lr, n_actions, name, input_dims, fc1_units, fc2_units, sess, batch_size):
        self.lr = lr
        self.n_actions = n_actions
        self.name = name
        self.fc1_units = fc1_units
        self.fc2_units = fc2_units
        self.sess = sess
        self.batch_size = batch_size
        with tf.variable_scope(self.name):
            self.input = tf.placeholder(tf.float32,
                                        shape=[None, *input_dims],
                                        name='inputs')
            self.advantage_estimates = tf.placeholder(tf.float32,
                                                      shape=[None, self.n_actions],
                                                      name='advantage_estimates')
            self.mu = self.build_network()
            self.params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                            scope=self.name)
            self.policy_gradients = self.calculate_gradients()
            self.optimizer = tf.train.AdamOptimizer(self.lr).apply_gradients(zip(self.policy_gradients, self.params))
        self.sess.run(tf.global_variables_initializer())

    def build_network(self):
        flat = tf.layers.flatten(self.input)
        dense1 = tf.layers.dense(flat, units=self.fc1_units,
                                 activation=tf.nn.relu)
        dense2 = tf.layers.dense(dense1, units=self.fc2_units,
                                 activation=tf.nn.relu)
        return tf.layers.dense(dense2, units=self.n_actions)

    def calculate_gradients(self, ):
        actor_gradients = []
        # TODO: can I do tf.math.log here directly?
        un_normalized_actor_gradients = tf.gradients(tf.math.log(self.mu), self.params, self.advantage_estimates)
        for g in un_normalized_actor_gradients:
            actor_gradients.append(tf.div(g, self.batch_size))
        return actor_gradients


class ValueNetwork(DQN):
    def __init__(self, lr, n_actions, name, input_dims, fc1_units, fc2_units, sess):
        super().__init__(lr, n_actions, name, input_dims, fc1_units, fc2_units, sess)
        with tf.variable_scope(self.name):
            self.v_value = self.q_values  # reusing the network and not causing confusion
            self.v_target = self.q_target # TODO: any side effect? e.g. params not updating


class VPGAgent:
    def __init__(self, policy_lr, value_lr, input_dims, n_actions=2,
                 mem_size=10000, fc1_units=64, fc2_units=32,
                 batch_size=64):
        self.experience_memory = TrajectoryStore(mem_size, input_dims, n_actions, action_discrete=False)
        self.batch_size = batch_size
        self.sess = tf.Session()
        self.policy = PolicyNetwork(policy_lr, n_actions, input_dims=input_dims, name='actor',
                                    fc1_units=fc1_units, fc2_units=fc2_units, sess=self.sess,
                                    batch_size=batch_size)
        self.value = ValueNetwork(value_lr, n_actions, input_dims=input_dims, name='target_actor',
                                  fc1_units=fc1_units, fc2_units=fc2_units, sess=self.sess)

    def record(self, state, action, reward, new_state, terminal):
        self.experience_memory.store(state, action, reward, new_state, terminal)

    def clear_store(self):
        self.experience_memory.clear()

    def act(self, state):
        state = state[np.newaxis, :]
        mu = self.sess.run(self.policy.mu, feed_dict={self.policy.input: state})
        return mu[0]

    def learn(self):
        states, actions, rewards, new_states, terminals = self.experience_memory.get_trajectory()

        rewards_to_go = self._calculate_rewards_to_go(rewards, terminals)
        advantage_estimates = self._calculate_advantage_estimates(states, rewards_to_go)

        self.sess.run(self.policy.optimizer,
                      feed_dict={
                          self.policy.advantage_estimates: advantage_estimates,
                          self.policy.input: states
                      })

        self.sess.run(self.value.optimizer,
                      feed_dict={
                          self.value.v_target: rewards_to_go,
                          self.value.input: states
                      })

    @staticmethod
    def _calculate_rewards_to_go(rewards, is_end):
        rewards_to_go = []
        terminates = np.where(is_end == 0)[0]
        for i, v in enumerate(rewards):
            if i > terminates[0]:
                terminates = terminates[1:]
            end = terminates[0]
            rewards_to_go.append(np.sum(rewards[i:end]))

        return np.reshape(rewards_to_go, (len(rewards_to_go), 1))

    def _calculate_advantage_estimates(self, states, rewards_to_go):
        # TODO: GAE, I am not sure how to do the estimation
        advantage_estimates = []
        v_curr = self.sess.run(self.value.v_value,
                               feed_dict={
                                   self.value.input: states
                               })
        for i, v in enumerate(rewards_to_go):
            advantage_estimates.append(v - v_curr[i])

        return np.reshape(advantage_estimates, (len(advantage_estimates), 1))
