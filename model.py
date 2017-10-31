import tensorflow as tf
from tf_utils import fc, get_trainable_variables, transfer_learning
from tensorflow.contrib.framework import get_or_create_global_step


class ActorCritic(object):
    def __init__(self, scope, obs_dim, acts_dim, network_config, target=None):
        self.scope = scope
        self.global_step = get_or_create_global_step()
        self.__init_ph(obs_dim=obs_dim)
        self.__build_graph(acts_dim=acts_dim, act=network_config['act'], units=network_config['units'])
        self.__loss_op(acts_dim=acts_dim, beta=network_config['beta'])
        self.__grads_op()
        if target is not None:
            self.__train_op(target, optim=network_config['optim'], clip=network_config['clip'],
                            lr=network_config['lr'])

    def __init_ph(self, obs_dim):
        self.obs = tf.placeholder(tf.float32, [None, obs_dim], name='obs')
        self.acts = tf.placeholder(tf.int32, [None], name='acts')
        self.rws = tf.placeholder(tf.float32, [None], name='rws')
        self.adv = tf.placeholder(tf.float32, [None], name='adv')

    @staticmethod
    def __build_lstm(obs, units=64):
        from tensorflow.contrib.rnn import BasicLSTMCell, LSTMStateTuple
        cell = BasicLSTMCell(num_units=units)
        c_in = tf.placeholder(dtype=tf.float32, shape=(1, cell.state_size.c), name='c_in')
        h_in = tf.placeholder(dtype=tf.float32, shape=(1, cell.state_size.h), name='h_in')
        state_in = LSTMStateTuple(c_in, h_in)
        h, state_out = tf.nn.dynamic_rnn(cell=cell, inputs=tf.expand_dims(obs, [0]), time_major=False,
                                         initial_state=state_in, sequence_length=tf.shape(obs)[:1])
        h = tf.reshape(h, [-1, cell.state_size.c])
        # c_out, h_out = state_out
        return state_in, state_out, h

    def __build_graph(self, acts_dim, units=(64), act=tf.nn.relu):
        self.cell_size = units
        with tf.variable_scope(self.scope):
            h = self.obs
            # for idx, size in enumerate(units):
            #     h = fc(x=h, h_size=size, act=act, name='h_{}'.format(idx))
            self.state_in, self.state_out, h = self.__build_lstm(obs=h, units=units)
            with tf.variable_scope('actor'):
                self.pi = fc(h, h_size=acts_dim, act=tf.nn.softmax, name='actor')
            with tf.variable_scope('critic'):
                v = fc(h, h_size=1, act=None, name='critic')
                self.v = tf.reshape(v, [-1])

        self.params = get_trainable_variables(scope=self.scope)

    def __grads_op(self):
        self.grads = tf.gradients(self.loss, self.params)

    def __loss_op(self, acts_dim, beta):
        self.vf_loss = tf.reduce_sum(tf.square(self.rws - self.v), name='vf_loss')
        self.entropy = -tf.reduce_sum(self.pi * tf.log(self.pi + 1e-5), axis=-1)  # encourage exploration
        log_pi = tf.reduce_sum(tf.log(self.pi) * tf.one_hot(self.acts, acts_dim, dtype=tf.float32), axis=-1)
        self.pi_loss = - tf.reduce_sum(log_pi * self.adv, name='pi_loss')

        self.loss = self.pi_loss + .5 * self.vf_loss - beta * self.entropy

    def __train_op(self, target, optim, clip, lr):
        self.sync_op = transfer_learning(to_tensors=self.params, from_tensors=target.params)
        self.grads, _ = tf.clip_by_global_norm(t_list=self.grads, clip_norm=clip)
        self.train_op = optim(lr).apply_gradients(
            zip(self.grads, target.params), global_step=self.global_step)
