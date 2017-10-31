from model import ActorCritic
from tensorflow import get_default_session
import numpy as np
from tf_utils import compute_gae


class A3C(object):
    def __init__(self, scope, obs_dim, acts_dim, target, network_config):
        self.model = ActorCritic(scope=scope, obs_dim=obs_dim, acts_dim=acts_dim, target=target,
                                 network_config=network_config)

    def train(self, feed_dict):  # run by a local
        sess = get_default_session()  # local grads applies to global net
        pi_loss, vf_loss, global_step, _ = sess.run(
            [self.model.pi_loss, self.model.vf_loss, self.model.global_step, self.model.train_op], feed_dict)
        return (pi_loss, vf_loss), global_step

    def reset(self):
        return [np.zeros((1, self.model.cell_size)), np.zeros((1, self.model.cell_size))]

    def sync_from_target(self):  # run by a local
        sess = get_default_session()
        sess.run([self.model.sync_op])

    def get_v(self, ob, state_in):
        sess = get_default_session()
        return sess.run(self.model.v, {self.model.obs: ob, self.model.state_in: state_in})

    def step(self, ob, state_in):  # run by a local
        sess = get_default_session()
        pi, v, s_out = sess.run([self.model.pi, self.model.v, self.model.state_out],
                                feed_dict={self.model.obs: [ob], self.model.state_in: state_in})  # ob[np.newaxis, :]})
        act = np.random.choice(pi.shape[1], p=pi.ravel())  # select action w.r.t the actions prob
        return act, v[0], s_out

    def get_batch(self, batch, ob1, done, state_in, gamma=.95, _lambda=1.):

        batch = np.array(batch)
        obs = batch[:, 0]
        acts = batch[:, 1]
        rws = batch[:, 2]
        vs = batch[:, 3]

        r_hat = 0
        if done == False:
            r_hat = self.get_v(ob=[ob1], state_in=state_in)[0]

        # d_rws, adv = compute_gae(rws =rws, vs= vs, r_hat=r_hat, gamma=gamma, _lambda=_lambda)
        d_rws = []
        for r in rws[::-1]:  # reverse buffer r
            r_hat = r + gamma * r_hat
            d_rws.append(r_hat)

        d_rws = np.array(d_rws)[::-1]  # .copy()
        adv = d_rws - vs

        feed_dict = {
            self.model.obs: np.vstack(obs),
            self.model.acts: acts,
            self.model.rws: d_rws,
            self.model.adv: adv,  # d_rws - np.array(vs)
            self.model.state_in: self.reset()
        }
        return feed_dict
