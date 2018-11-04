
from softqlearning.algorithms.sql import SQL
from softqlearning.misc.kernel import adaptive_isotropic_gaussian_kernel
from softqlearning.replay_buffers.simple_replay_buffer import SimpleReplayBuffer
from softqlearning.value_functions.value_function import NNQFunction
from softqlearning.policies.stochastic_policy import StochasticNNPolicy
from softqlearning.misc.sampler import SimpleSampler
from softqlearning.environment.space_robot_3link import SpaceRobot3link
# from softqlearning.environment.space_robot_7links import SpaceRobot3link
# from softqlearning.environment.space_robot_double_3links import SpaceRobot3link

from experiment.hyper_parameters import SHARED_PARAMS, ENV_PARAMS

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


env = SpaceRobot3link()

pool = SimpleReplayBuffer(
    env_spec=env.spec, max_replay_buffer_size=SHARED_PARAMS['max_pool_size'])

sampler = SimpleSampler(
    max_path_length=ENV_PARAMS['max_path_length'],
    min_pool_size=SHARED_PARAMS['min_pool_size'],
    batch_size=SHARED_PARAMS['batch_size'])

base_kwargs = dict(
    epoch_length=SHARED_PARAMS['epoch_length'],
    n_epochs=ENV_PARAMS['n_epochs'],
    n_train_repeat=SHARED_PARAMS['n_train_repeat'],
    eval_render=False,
    eval_n_episodes=1,
    sampler=sampler)

qf = NNQFunction(env_spec=env.spec, hidden_layer_sizes=SHARED_PARAMS['Q_layer_size'])

policy = StochasticNNPolicy(env_spec=env.spec, hidden_layer_sizes=SHARED_PARAMS['policy_layer_size'],
                            squash=ENV_PARAMS['squash'])

algorithm = SQL(
    base_kwargs=base_kwargs,    env=env,
    pool=pool,
    qf=qf,
    policy=policy,
    kernel_fn=adaptive_isotropic_gaussian_kernel,
    kernel_n_particles=SHARED_PARAMS['kernel_particles'],
    kernel_update_ratio=SHARED_PARAMS['kernel_update_ratio'],
    value_n_particles=SHARED_PARAMS['value_n_particles'],
    td_target_update_interval=SHARED_PARAMS['td_target_update_interval'],
    qf_lr=SHARED_PARAMS['qf_lr'],
    policy_lr=SHARED_PARAMS['policy_lr'],
    discount=SHARED_PARAMS['discount'],
    reward_scale=ENV_PARAMS['reward_scale'],
    save_full_state=False)

load_path = SHARED_PARAMS['model_save_path'] + 'stage1' + '/model-500.ckpt'
algorithm.train(load=load_path)
