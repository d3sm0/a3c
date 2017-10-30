import multiprocessing
from tf_utils import discount
import threading
import tensorflow as tf
import numpy as np
import gym
from tf_utils import fc, p_to_q, build_z, get_pa

GAME = 'CartPole-v0'
OUTPUT_GRAPH = True
LOG_DIR = './log'
N_WORKERS = 1  # multiprocessing.cpu_count()
MAX_GLOBAL_EP = 1000
GLOBAL_NET_SCOPE = 'target'
UPDATE_GLOBAL_ITER = 10
GAMMA = 0.9
ENTROPY_BETA = 0.001
LR_A = 0.001  # learning rate for actor
LR_C = 0.001  # learning rate for critic
GLOBAL_RUNNING_R = []
GLOBAL_EP = 0
UNITS = [200]
env = gym.make(GAME)
obs_dim = env.observation_space.shape[0]
acts_dim = env.action_space.n
n_atoms = 11
v_min = -10.
v_max = 10.

class ActorCritic(object):
    def __init__(self, scope, obs_dim, acts_dim, reuse=False, target=None):
        self.scope = scope
        self.reuse_params = reuse
        with tf.variable_scope(scope):
            self.__init_ph()
            self.__build_graph()
            self.__predict_op()
            self.__build_categorical(v_min, v_max, n_atoms, gamma = .99)
            self.__loss_op()
            self.__train_op()
            if target is not None:
                self.__sync_op(target)

    def __init_ph(self):
        self.obs = tf.placeholder(tf.float32, [None, obs_dim], name='obs')
        self.acts = tf.placeholder(tf.int32, [None, ], name='acts')
        self.rws = tf.placeholder(tf.float32, [None, ], name='rws')
        self.adv = tf.placeholder(tf.float32, [None, ], name='adv')
        self.dones = tf.placeholder(tf.float32, [None, ], name='dones')
        self.thtz = tf.placeholder(dtype=tf.float32, shape=[None, n_atoms], name='T_pi')

    def __build_graph(self):
        h = self.obs
        # for idx, size in enumerate(UNITS):
        #     h = fc(x=h, h_size = size, act = tf.nn.relu, name = 'h_{}'.format(idx), reuse=self.reuse_params)

        with tf.variable_scope('actor'):
            h = fc(x=self.obs, h_size=200, act=tf.nn.relu, name='h_pi')
            self.pi = fc(h, h_size=acts_dim, act=tf.nn.softmax, name='actor')
        with tf.variable_scope('critic'):
            h = fc(x=self.obs, h_size=200, act=tf.nn.relu, name='h_vf')
            logits = fc(x =h, h_size  = acts_dim * n_atoms, act=None, name = 'q_sa')
            logits = tf.reshape(logits, shape = (-1, acts_dim, n_atoms))
            self.p = tf.nn.softmax(logits=logits, dim = -1)

            # v = fc(h, h_size=1, act=None, name='critic')
            # self.v = tf.reshape(v, [-1])


        #
        # w_init = tf.random_normal_initializer(0., .1)
        # with tf.variable_scope('actor'):
        #     l_a = tf.layers.dense(self.obs, 200, tf.nn.relu6, kernel_initializer=w_init, name='la')
        #     self.pi = tf.layers.dense(l_a, acts_dim, tf.nn.softmax, kernel_initializer=w_init, name='ap')
        # with tf.variable_scope('critic'):
        #     l_c = tf.layers.dense(self.obs, 100, tf.nn.relu6, kernel_initializer=w_init, name='lc')
        #     v = tf.layers.dense(l_c, 1, kernel_initializer=w_init, name='v')  # state value
        #     self.v = tf.reshape(v, shape=[-1])
        # self.params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope)
        self.a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope + '/actor')
        self.c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope + '/critic')


    def __train_op(self):
        with tf.name_scope('local_grad'):
            self.a_grads = tf.gradients(self.pi_loss, self.a_params)
            self.c_grads = tf.gradients(self.vf_loss, self.c_params)
            # self.grads = tf.gradients(self.loss, self.params)

    def __loss_op(self, beta = .01):
        # td = tf.subtract(self.rws, self.v, name='TD_error')
        with tf.name_scope('c_loss'):
            # self.vf_loss = tf.reduce_sum(tf.square(td))
            batch_size = tf.shape(self.obs)[0]
            p_target = get_pa(p=self.p, acts=self.acts, batch_size=batch_size)
            # cross entropy
            self.vf_loss = tf.reduce_mean(tf.reduce_sum(-self.ThTz * tf.log(p_target  + 1e-5), axis=-1),
                                          name='empirical_cross_entropy')

        with tf.name_scope('a_loss'):
            log_prob = tf.reduce_sum(tf.log(self.pi) * tf.one_hot(self.acts, acts_dim, dtype=tf.float32),
                                     axis=1, keep_dims=True)
            exp_v = log_prob * self.ThTz #  self.adv  # td
            entropy = -tf.reduce_sum(self.pi * tf.log(self.pi + 1e-5),
                                     axis=1, keep_dims=True)  # encourage exploration
            self.exp_v = beta * entropy + exp_v
            self.pi_loss = tf.reduce_sum(-self.exp_v)

        # with tf.name_scope('loss'):
        #     self.loss = self.pi_loss + .5 * self.vf_loss

    def __sync_op(self, target):
        with tf.name_scope('sync'):
            with tf.name_scope('pull'):
                self.sync_actor_op = [l_p.assign(g_p) for l_p, g_p in zip(self.a_params, target.a_params)]
                self.sync_critic_op = [l_p.assign(g_p) for l_p, g_p in zip(self.c_params, target.c_params)]
                # self.sync_op = [l_p.assign(g_p) for l_p, g_p in zip(self.params, target.params)]
            with tf.name_scope('push'):
                self.train_actor_op = tf.train.RMSPropOptimizer(learning_rate=LR_A).apply_gradients(
                    zip(self.a_grads, target.a_params))
                self.train_critic_op = tf.train.RMSPropOptimizer(learning_rate=LR_C).apply_gradients(
                    zip(self.c_grads, target.c_params))
                #
                # self.train_op = tf.train.RMSPropOptimizer(learning_rate=LR_A).apply_gradients(
                #     zip(self.grads, target.params)
                # )

    def __predict_op(self):
        self.q_values  = p_to_q(self.p, v_min, v_max, n_atoms)
        self.next_action = tf.argmax(self.q_values, axis = 1, output_type=tf.int32)
    def __build_categorical(self, v_min, v_max, n_atoms, gamma):
        z, dz = build_z(v_min=v_min, v_max=v_max, n_atoms=n_atoms)
        batch_size = tf.shape(self.rws)[0]
        with tf.variable_scope('categorical'):
            # we need to build a categorical index for the bin. Thus we get the index, concat with the action which is of size
            # (batch_size,), reshape it of (2,batch_size) and transpose it. The final matrix is of size (batch_size,2) where the first columns has index and
            # seocond column has action
            # get probs of selected actions. p is of shape (batch_size, action_size, dict_size)
            # p_best (batch_size, nb_atoms)
            self.p_best = get_pa(p=self.p, acts=self.next_action, batch_size=batch_size)
            # replicates z, batch_size times, this is building the integrations support over the atom dimension
            Z = tf.reshape(tf.tile(z, [batch_size]), shape=[batch_size, n_atoms])
            # replicates rws (batch_size, ) over n_atoms, reshape it in n_atoms, batch_size, and traspose it. Final dim (batch_size, n_atoms)
            R = tf.transpose(tf.reshape(tf.tile(self.rws, [n_atoms]), shape=[n_atoms, batch_size]))
            # Apply bellman operator over the Z random variable
            Tz = tf.clip_by_value(R + gamma * tf.einsum('ij,i->ij', Z, 1. - self.dones), clip_value_min=v_min,
                                  clip_value_max=v_max)
            # Expanded over in the atom_dimension. Batch_size, n_atoms**2
            Tz = tf.reshape(tf.tile(Tz, [1, n_atoms]), (-1, n_atoms, n_atoms))
            Z = tf.reshape(tf.tile(Z, [1, n_atoms]), shape=(-1, n_atoms, n_atoms))
            # Rescale the bellman operator over the support of the Z random variable
            Tzz = tf.abs(Tz - tf.transpose(Z, perm=(0, 2, 1))) / dz
            Thz = tf.clip_by_value(1 - Tzz, 0, 1)
            # Integrate out the k column of the atom dimension
            self.ThTz = tf.einsum('ijk,ik->ij', Thz, self.p_best)


class A3C(object):
    def __init__(self, scope, obs_dim, acts_dim, target):
        self.model = ActorCritic(scope=scope, obs_dim=obs_dim, acts_dim=acts_dim, target=target)
    def get_v(self, ob):
        sess = tf.get_default_session()
        v = sess.run(self.model.q_values, feed_dict = {self.model.obs:ob})
        # exp value of V(s) under pi, starting from the integral of the Q(s,a) over all a
        # this is most likely wrong
        return np.mean(np.sum(v, axis = 1))

    def train(self, feed_dict):  # run by a local
        sess = tf.get_default_session()
        pi_loss, vf_loss,  _,_ = sess.run([self.model.pi_loss,
                                         self.model.vf_loss,
                                         # self.model.vf_loss,
                                          self.model.train_actor_op,
                                          self.model.train_critic_op
                                          # self.model.train_op
                                         ],
                                         feed_dict)  # local grads applies to global net
        return pi_loss, vf_loss

    def sync_from_target(self):  # run by a local
        sess = tf.get_default_session()
        sess.run([
            # self.model.sync_op
            self.model.sync_actor_op,
            self.model.sync_critic_op
        ])

    def step(self, ob):  # run by a local
        sess = tf.get_default_session()
        pi, v = sess.run([self.model.pi, self.model.p], feed_dict={self.model.obs: [ob]}) # ob[np.newaxis, :]})
        action = np.random.choice(pi.shape[1], p=pi.ravel())  # select action w.r.t the actions prob
        v = v[0][action].mean()
        return action, v

    def get_batch(self, batch, ob1, done, gamma=.99, _lambda=1.):

        # sess = tf.get_default_session()
        batch = np.array(batch).copy()
        obs = batch[:, 0]
        acts = batch[:, 1]
        rws = batch[:, 2]
        vs = batch[:, 3]
        ds = batch[:, 4]
        # obs1 = np.append(obs[1:], ob1)

        r_hat = 0
        # r_hat is 1 x atoms
        if done == False:
            # r_hat = sess.run(self.model.v, {self.model.obs: [ob1]})[0]  # ob1[np.newaxis, :]})[0]  # [0, 0]
            r_hat = self.get_v(ob =[ob1])
        d_rws = []
        for r in rws[::-1]:  # reverse buffer r
            r_hat = r + GAMMA * r_hat
            d_rws.append(r_hat)

        d_rws = np.array(d_rws)[::-1]  # .copy()
        adv = d_rws - vs
        # obs, acts, d_rws = np.vstack(obs), np.array(acts), np.array(
        #     d_rws)

        # d_rws = np.append(rws, r_hat)
        # d_rws = discount(d_rws, GAMMA)[:-1]
        # vs = np.append(vs, r_hat)
        # td_error = rws + GAMMA * vs[1:] - vs[:-1]
        # adv = discount(td_error, gamma=GAMMA * _lambda)
        feed_dict = {
            self.model.obs: np.vstack(obs),
            self.model.acts: acts,
            self.model.rws: rws,
            self.model.dones:ds,
            self.model.adv: adv  # d_rws - np.array(vs)
        }
        losses = self.train(feed_dict)
        self.sync_from_target()
        return losses


class Worker(object):
    def __init__(self, name, target):
        self.env = gym.make(GAME)  # .unwrapped
        self.name = name
        self.agent = A3C(name, env.observation_space.shape[0], env.action_space.n, target)
        self.ep_stats = {'ep_rw': 0, 'ep_len': 0, 'total_ep': 0}

    def run(self, sess, coord):

        with sess.as_default():
            global GLOBAL_RUNNING_R, GLOBAL_EP
            total_step = 1
            batch = []
            while not coord.should_stop() and GLOBAL_EP < MAX_GLOBAL_EP:
                ob = self.env.reset()
                ep_r = 0
                while True:
                    # if self.name == 'W_0':
                        # self.env.render()
                    act, v = self.agent.step(ob)
                    ob1, r, done, info = self.env.step(act)
                    ep_r += r
                    batch.append((ob, act, r, v, done))
                    if total_step % UPDATE_GLOBAL_ITER == 0 or done:  # update global and assign to local net
                        losses = self.agent.get_batch(batch=batch, ob1=ob1, done=done)
                        batch = []
                    ob = ob1
                    total_step += 1
                    if done:
                        print(
                            self.name,
                            "Ep:", GLOBAL_EP,
                            "| Ep_r: %i" % ep_r,
                            '| Ep_a_loss: %i' % losses[0],
                            '| Ep_c_Loss: %i' % losses[1]
                        )
                        GLOBAL_EP += 1
                        break


def main():
    sess = tf.Session()
    with tf.device("/cpu:0"):
        target = ActorCritic(scope='target', obs_dim=obs_dim, acts_dim=acts_dim)  # we only need its params
        workers = []
        # Create worker
        for i in range(N_WORKERS):
            i_name = 'W_%i' % i  # worker name
            workers.append(Worker(i_name, target))
    coord = tf.train.Coordinator()
    sess.run(tf.global_variables_initializer())
    worker_threads = []
    for worker in workers:
        t = threading.Thread(target=worker.run, args=(sess, coord,))
        t.start()
        worker_threads.append(t)
    coord.join(worker_threads)
    # plt.plot(np.arange(len(GLOBAL_RUNNING_R)), GLOBAL_RUNNING_R)
    # plt.xlabel('step')
    # plt.ylabel('Total moving reward')
    # plt.show()


if __name__ == "__main__":
    main()
