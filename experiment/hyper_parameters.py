SHARED_PARAMS = {
    'policy_lr': 1E-4,
    'qf_lr': 1E-4,
    'discount': 0.99,
    'Q_layer_size': [128, 128],
    'policy_layer_size': [200, 200],
    'batch_size': 256,
    'max_pool_size': 1E6,
    'min_pool_size': 1000,
    'epoch_length': 1000,
    'pkl_gap': 5,
    'n_train_repeat': 1,
    'kernel_particles': 32,
    'kernel_update_ratio': 0.5,
    'value_n_particles': 32,
    'td_target_update_interval': 1000,
    'model_save_path': '../data',
    'viz_log_path': '../viz_data',
    }


ENV_PARAMS = {
    'max_path_length': 200,
    'n_epochs': 500,
    'reward_scale': 30,
    'final_phase_reward_scaler': 1.5,
    'reward_factor_w1': 1E-3,
    'reward_factor_wlog': 1.0,
    'reward_factor_w2': 0.01,
    'alpha': 1E-6,
    'distance_tolerance': 0.04,
    'torques_norm_tolerance': 0.1,
    'max_torque': 2,
    'squash': True
    }

AVAILABLE_ENVS = ['SpaceRobot3link', 'SpaceRobotDouble3link']
DEFAULT_ENV = 'SpaceRobot3link'
