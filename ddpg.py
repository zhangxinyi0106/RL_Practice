import tensorflow as tf
import numpy as np

from dqn import DQN, ExperienceMemory, NatureDQNAgent


class OUActionNoise(object):
    """This is to generate OU Noise, Copied from the Internet
    """
    def __init__(self, mu, sigma=0.15, theta=.2, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.x_prev = np.zeros_like(self.mu)

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x


class CriticNetwork:
    def __init__(self, lr, n_actions, name, input_dims, fc1_units, fc2_units, sess):
        # TODO: repeated with DQN, can implement a more abstract model
        self.lr = lr
        self.n_actions = n_actions
        self.name = name
        self.fc1_units = fc1_units
        self.fc2_units = fc2_units
        self.sess = sess
        with tf.variable_scope(self.name):
            self.input = tf.placeholder(tf.float32,
                                        shape=[None, *input_dims],
                                        name='inputs')
            self.actions = tf.placeholder(tf.float32,
                                          shape=[None, self.n_actions],
                                          name='actions')
            self.q_values = self.build_network()
            self.q_target = tf.placeholder(tf.float32,
                                           shape=[None, self.n_actions],
                                           name='q_target')
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_values, self.q_target))
            self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
        self.sess.run(tf.global_variables_initializer())
        self.params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                        scope=self.name)
        self.action_gradients = tf.gradients(self.q_values, self.actions)

    def build_network(self):
        flat = tf.layers.flatten(self.input)
        dense1_state = tf.layers.dense(flat, units=self.fc1_units,
                                       activation=tf.nn.relu)
        dense1_action = tf.layers.dense(self.actions, units=self.fc1_units,
                                        activation=tf.nn.relu)
        state_actions_joint = tf.add(dense1_state, dense1_action)
        state_actions_joint = tf.nn.relu(state_actions_joint)
        dense2 = tf.layers.dense(state_actions_joint, units=self.fc2_units,
                                 activation=tf.nn.relu)
        return tf.layers.dense(dense2, units=self.n_actions)

    def get_action_grads(self, inputs, actions):
        return self.sess.run(self.action_gradients,
                             feed_dict={self.input: inputs,
                                        self.actions: actions})


class ActorNetwork(DQN):
    """Just for convenience, reusing the DQN network
    """
    def __init__(self, lr, n_actions, name, input_dims, fc1_units,
                 fc2_units, sess, batch_size):
        super().__init__(lr, n_actions, name, input_dims, fc1_units,
                         fc2_units, sess)
        self.batch_size = batch_size
        with tf.variable_scope(self.name):
            self.action_gradient = tf.placeholder(tf.float32,
                                                  shape=[None, self.n_actions],
                                                  name='gradients')
            self.mu = self.q_values  # reusing the network and not causing confusion
        self.actor_gradients = self.calculate_gradients()
        self.optimizer = tf.train.AdamOptimizer(self.lr).apply_gradients(zip(self.actor_gradients, self.params))

    def calculate_gradients(self):
        actor_gradients = []
        # calculate ∇aQ(s, a|θQ)|s=si,a=μ(si)∇θμ μ(s|θμ)|si
        # put a minus sign here in order to turn minimize to maximize later
        un_normalized_actor_gradients = tf.gradients(self.mu, self.params, -self.action_gradient)
        for g in un_normalized_actor_gradients:
            # calculate 1/N(∇aQ(s, a|θQ)|s=si,a=μ(si)∇θμ μ(s|θμ)|si)
            actor_gradients.append(tf.div(g, self.batch_size))
        return actor_gradients


class DDPGAgent:
    def __init__(self, actor_lr, critic_lr, input_dims, tau, gamma=0.99, n_actions=2,
                 mem_size=10000, fc1_units=64, fc2_units=32,
                 batch_size=64):
        self.gamma = gamma
        self.tau = tau
        self.experience_memory = ExperienceMemory(mem_size, input_dims, n_actions, action_discrete=False)
        self.batch_size = batch_size
        self.sess = tf.Session()
        self.actor = ActorNetwork(actor_lr, n_actions, input_dims=input_dims, name='actor',
                                  fc1_units=fc1_units, fc2_units=fc2_units, sess=self.sess,
                                  batch_size=batch_size)
        self.target_actor = ActorNetwork(actor_lr, n_actions, input_dims=input_dims, name='target_actor',
                                         fc1_units=fc1_units, fc2_units=fc2_units, sess=self.sess,
                                         batch_size=batch_size)
        self.update_target_actor(init=True)
        self.critic = CriticNetwork(critic_lr, n_actions, input_dims=input_dims, name='critic',
                                    fc1_units=fc1_units, fc2_units=fc2_units, sess=self.sess)
        self.target_critic = CriticNetwork(critic_lr, n_actions, input_dims=input_dims, name='target_critic',
                                           fc1_units=fc1_units, fc2_units=fc2_units, sess=self.sess)
        self.update_target_critic(init=True)
        self.noise = OUActionNoise(mu=np.zeros(n_actions))

    def update_target_critic(self, init=False):
        self.update_target_network_params(self.target_critic, self.critic, init)

    def update_target_actor(self, init=False):
        self.update_target_network_params(self.target_actor, self.actor, init)

    def update_target_network_params(self, target_network, network, init):
        target_params = target_network.params
        params = network.params

        # TODO: Not a good idea to put a for loop here
        for t, e in zip(target_params, params):
            if not init:
                e = tf.multiply(e, self.tau) + tf.multiply(t, 1 - self.tau)
            self.sess.run(tf.assign(t, e))

    def record(self, state, action, reward, new_state, terminal):
        self.experience_memory.store(state, action, reward, new_state, terminal)

    def act(self, state, batch=False, test=False):
        # env will clip the act, not handled here
        if not batch:
            state = state[np.newaxis, :]
        mu = self.sess.run(self.actor.mu, feed_dict={self.actor.input: state})

        if test:
            return mu if batch else mu[0]

        noise = self.noise()
        mu_prime = mu + noise
        return mu_prime if batch else mu[0]

    def learn(self):
        if self.experience_memory.mem_counter < self.batch_size:
            return

        states, actions, rewards, new_states, terminals = self.experience_memory.sample(self.batch_size)

        # μ′(si+1|θμ′ )
        a_target = self.sess.run(self.target_actor.mu,
                                 feed_dict={
                                     self.target_actor.input: new_states,
                                 })

        # Q′(si+1, μ′(si+1|θμ′ )|θQ′ )
        q_next = self.sess.run(self.target_critic.q_values,
                               feed_dict={
                                   self.target_critic.input: new_states,
                                   self.target_critic.actions: a_target,
                               })

        # yi = ri + γQ′(si+1, μ′(si+1|θμ′ )|θQ′ )
        rewards = rewards.reshape(self.batch_size, 1)
        terminals = terminals.reshape(self.batch_size, 1)
        q_target = np.add(rewards, self.gamma * q_next * terminals)

        # Update critic by minimizing the loss:L=1/N*sum((yi−Q(si,ai|θQ))2)
        self.sess.run(self.critic.optimizer,
                      feed_dict={self.critic.input: states,
                                 self.critic.actions: actions,
                                 self.critic.q_target: q_target})
        # a=μ(si)
        latest_a = self.act(states, batch=True, test=True)

        # calculate  ∇aQ(s, a|θQ)|s=si,a=μ(si)
        grads = self.critic.get_action_grads(states, latest_a)

        self.sess.run(self.actor.optimizer,
                      feed_dict={self.actor.input: states,
                                 self.actor.action_gradient: grads[0]})

        # TODO: These two steps are taking extremely long time
        # Do we really need to update them every step as stated in paper?
        self.update_target_critic()
        self.update_target_actor()
