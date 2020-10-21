import tensorflow as tf
import numpy as np
import argparse
import time
import os

from softqlearning.misc.nn import feedforward_net
from softqlearning.environment.space_robot_3link import SpaceRobot3link
from softqlearning.environment.space_robot_double_3links import SpaceRobotDouble3link
from experiment.hyper_parameters import SHARED_PARAMS, ENV_PARAMS, AVAILABLE_ENVS, DEFAULT_ENV


viz_log_path = SHARED_PARAMS['viz_log_path']
if not os.path.exists(viz_log_path):
    os.mkdir(viz_log_path)

parser = argparse.ArgumentParser(description='Train control policies.')
parser.add_argument('--env', type=str, choices=AVAILABLE_ENVS, default=DEFAULT_ENV)
parser.add_argument('--model', type=str, default='')
arg_parser = parser.parse_args()

if arg_parser.env == 'SpaceRobot3link':
    env = SpaceRobot3link()
elif arg_parser.env == 'SpaceRobotDouble3link':
    env = SpaceRobotDouble3link
else:
    raise NotImplementedError

action_dim = env.action_dims
observation_dim = env.obs_dims
squash = ENV_PARAMS['squash']
layer_size = SHARED_PARAMS['policy_layer_size'] + [action_dim]


def controller(obs):
    """
    trained policy / controller
    :param obs: s_t [1, observation_dim]
    :return: a_t [1, action_dim]
    """
    single_latent = np.random.normal(size=(1, action_dim))

    raw_action = sess.run(raw_actions, feed_dict={observation_ph: obs,
                                                  latent_ph: single_latent})
    tan_action = sess.run(tan_actions, feed_dict={observation_ph: obs,
                                                  latent_ph: single_latent})

    return tan_action if squash else raw_action


# stochastic policy network
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
    saver.restore(sess, arg_parser.model)

    # TODO: roll-out with the trained policy in V-REP
    observation = np.expand_dims(env.reset(), axis=0)       # s_0, shape[1, 18]

    for t in range(ENV_PARAMS['max_path_length']):
        action = controller(observation)        # action shape[1, 3]
        with open(viz_log_path + '/torques.txt', 'a+') as f:
            for a_t in action[0]:
                f.write(str(a_t) + ' ')
            f.write('\n')

        next_obs, reward, terminal, env_info = env.step(*action)        # *action, shape[3]
        with open(viz_log_path + '/angle.txt', 'a+') as f:
            for s_t in next_obs[0: 3]:
                f.write(str(s_t) + ' ')
            f.write('\n')
        with open(viz_log_path + '/angle_velocity.txt', 'a+') as f:
            for s_t in next_obs[3: 6]:
                f.write(str(s_t) + ' ')
            f.write('\n')
        with open(viz_log_path + '/reward_t.txt', 'a+') as f:
            f.write(str(reward) + '\n')

        observation = np.expand_dims(next_obs, axis=0)      # next_obs, shape[3] / observation, shape[1, 3]

        if terminal:
            for i in range(5):
                env.grip()
                time.sleep(1)
            break

    env.terminate()
