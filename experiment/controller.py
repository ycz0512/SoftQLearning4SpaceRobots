import tensorflow as tf
import numpy as np
import time

from softqlearning.misc.nn import feedforward_net
from softqlearning.environment.space_robot_3link import SpaceRobot3link
# from softqlearning.environment.space_robot_7links import SpaceRobot3link
# from softqlearning.environment.space_robot_double_3links import SpaceRobot3link
from experiment.hyper_parameters import SHARED_PARAMS, ENV_PARAMS

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

squash = ENV_PARAMS['squash']

env = SpaceRobot3link()
action_dim = env.action_dims
observation_dim = env.obs_dims

layer_size = SHARED_PARAMS['policy_layer_size'] + [action_dim]


def controller(observation):
    """
    trained policy / controller
    :param observation: s_t [1, observation_dim]
    :return: a_t [1, action_dim]
    """
    single_latent = np.random.normal(size=(1, action_dim))

    raw_action = sess.run(raw_actions, feed_dict={observation_ph: observation,
                                                  latent_ph: single_latent})
    tan_action = sess.run(tan_actions, feed_dict={observation_ph: observation,
                                                  latent_ph: single_latent})

    return tan_action if squash else raw_action


# forward computation graph of stochastic policy network
with tf.variable_scope('policy', reuse=False):
    observation_ph = tf.placeholder(tf.float32, shape=[None, observation_dim])
    latent_ph = tf.placeholder(tf.float32, shape=[None, action_dim])

    raw_actions = feedforward_net((observation_ph, latent_ph), layer_sizes=layer_size,
                                  activation_fn=tf.nn.relu, output_nonlinearity=None)
    tan_actions = tf.tanh(raw_actions)


saver = tf.train.Saver()
with tf.Session() as sess:

    # restore the trained model
    ckpt = tf.train.get_checkpoint_state(SHARED_PARAMS['model_save_path'])
    # saver.restore(sess, ckpt.model_checkpoint_path)
    saver.restore(sess, SHARED_PARAMS['model_save_path']+'model-465.ckpt')

    # TODO: roll-out with the trained policy
    observation = np.expand_dims(env.reset(), axis=0)       # s_0, shape[1, 18]

    for t in range(ENV_PARAMS['max_path_length']):
        action = controller(observation)        # action shape[1, 3]
        with open('../torques.txt', 'a+') as f:
            for a_t in action[0]:
                f.write(str(a_t) + ' ')
            f.write('\n')

        next_obs, reward, terminal, env_info = env.step(*action)        # *action, shape[3]
        with open('../angle.txt', 'a+') as f:
            for s_t in next_obs[0: 3]:
                f.write(str(s_t) + ' ')
            f.write('\n')
        with open('../angle_velocity.txt', 'a+') as f:
            for s_t in next_obs[3: 6]:
                f.write(str(s_t) + ' ')
            f.write('\n')
        with open('../reward_t.txt', 'a+') as f:
            f.write(str(reward) + '\n')

        observation = np.expand_dims(next_obs, axis=0)      # next_obs, shape[3] / observation, shape[1, 3]

        if terminal:
            for i in range(5):
                env.grip()
                time.sleep(1)
            break

    # for i in range(5):
    #     env.grip1()
    #     env.grip2()
    #     time.sleep(1)

    env.terminate()
