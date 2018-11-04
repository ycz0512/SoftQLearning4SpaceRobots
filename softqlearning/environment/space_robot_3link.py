import vrep
import numpy as np

from experiment.hyper_parameters import ENV_PARAMS


class BundleType(object):
    """
    This class bundles many fields, similar to a record or a mutable
    namedtuple.
    """
    def __init__(self, variables):
        for var, val in variables.items():
            object.__setattr__(self, var, val)

    # Freeze fields so new ones cannot be set.
    def __setattr__(self, key, value):
        if not hasattr(self, key):
            raise AttributeError("%r has no attribute %s" % (self, key))
        object.__setattr__(self, key, value)


class EnvSpec(BundleType):
    """ Collection of iteration variables. """
    def __init__(self):
        variables = {
            'action_space': None,       # object.
            'observation_space': None,  # object.
        }
        BundleType.__init__(self, variables)


class Dimension(BundleType):
    """ Collection of iteration variables. """
    def __init__(self):
        variables = {
            'flat_dim': None,       # object.
        }
        BundleType.__init__(self, variables)


class SpaceRobot3link(object):

    def __init__(self):
        self.handles = {}
        self.action_dims = 3
        self.obs_dims = 18

        self.env_spec = EnvSpec()
        self.env_spec.action_space = Dimension()
        self.env_spec.observation_space = Dimension()
        self.env_spec.action_space.flat_dim = self.action_dims
        self.env_spec.observation_space.flat_dim = self.obs_dims

        self.joint_target_velocities = np.ones(self.action_dims) * 10000     # make it gigantic enough to be the torque mode
        self.cur_obs = None         # current observation s_t
        self.num_episode = 1
        self.t = 1                          # the current time t for a trajectory.

        self._max_episode_length = ENV_PARAMS['max_path_length']        # t = {0, 1, ..., T}, T < max_episode_length.
        self.w1 = ENV_PARAMS['reward_factor_w1']
        self.w_log = ENV_PARAMS['reward_factor_wlog']
        self.w2 = ENV_PARAMS['reward_factor_w2']
        self.alpha = ENV_PARAMS['alpha']
        self.tol_1 = ENV_PARAMS['distance_tolerance']
        self.tol_2 = ENV_PARAMS['torques_norm_tolerance']
        self._max_torque = ENV_PARAMS['max_torque']
        self.final_phase_scaler = ENV_PARAMS['final_phase_reward_scaler']

    @property
    def spec(self):
        return self.env_spec

    def _init_vrep(self):       # get clientID
        vrep.simxFinish(-1)         # end all running communication threads
        clientID = vrep.simxStart('127.0.0.1', 19997, True, False, 5000, 0)       # get the clientID to communicate with the current V-REP
        self.clientID = clientID

        if self.clientID != -1:         # clientID = -1 means connection failed.(time-out)
            print('\n' + 'Connected to remote V-REP server.')
        else:
            print('\n' + 'Connection time-out !')
            raise Exception('Connection Failed !')

        vrep.simxSynchronous(self.clientID, True)       # enable synchronous operation to V-REP

    def _get_handles(self):       # get handles of objects in V-REP
        joint_names = ['Joint_1', 'Joint_2', 'Joint_3', 'Gjoint_1', 'Gjoint_2']     # object joint names in V-REP
        joint_handles = [vrep.simxGetObjectHandle(self.clientID, joint_names[i], vrep.simx_opmode_blocking)[1]
                         for i in range(len(joint_names))]         # retrieve object joint handles based on its name as a list

        base_handles = [vrep.simxGetObjectHandle(self.clientID, 'Base', vrep.simx_opmode_blocking)[1]]

        point_names = ['EE_point', 'EE_target_point']      # object EE_point names in V-REP
        point_handles = [vrep.simxGetObjectHandle(self.clientID, point, vrep.simx_opmode_blocking)[1]
                         for point in point_names]      # retrieve object EE_point handles as a list based on its name

        # handles dict
        self.handles['joint'] = joint_handles
        self.handles['base'] = base_handles
        self.handles['points'] = point_handles

    def _configure_initial_state(self, condition=0):
        # TODO: if we want to train the policy and Q-function under different initial configurations.
        pass

    def _get_observation(self):     # get current observations s_t
        state = np.zeros(self.obs_dims)
        # state is [θ1, θ2, θ3, ω1, ω2, ω3, Base_α, Base_β, Base_γ,
        #           Base_x, Base_y, Base_z, Vx, Vy, Vz, dα, dβ, dγ]

        Joint_angle = [vrep.simxGetJointPosition(self.clientID, self.handles['joint'][i], vrep.simx_opmode_blocking)[1]
                       for i in range(3)]      # retrieve the rotation angle as [θ1, θ2, θ3]
        state[0: 3] = Joint_angle

        Joint_velocity = [vrep.simxGetObjectFloatParameter(self.clientID, self.handles['joint'][i], 2012, vrep.simx_opmode_blocking)[1]
                          for i in range(3)]        # retrieve the joint velocity as [ω1, ω2, ω3]
        state[3: 6] = Joint_velocity

        Base_pose = [vrep.simxGetObjectOrientation(self.clientID, self.handles['base'][0],
                                                   -1, vrep.simx_opmode_blocking)[1]]
        state[6: 9] = Base_pose[0]          # retrieve the orientation of Base as [Base_α, Base_β, Base_γ]

        Base_position = [vrep.simxGetObjectPosition(self.clientID, self.handles['base'][0],
                                                    -1, vrep.simx_opmode_blocking)[1]]
        state[9: 12] = Base_position[0]     # retrieve the position of Base as [Base_x, Base_y, Base_z]

        _, Base_Vel, Base_Ang_Vel = vrep.simxGetObjectVelocity(self.clientID, self.handles['base'][0],
                                                               vrep.simx_opmode_blocking)
        state[12: 15] = Base_Vel        # retrieve the linear velocity of Base as [Vx, Vy, Vz]
        state[15: 18] = Base_Ang_Vel        # retrieve the angular velocity of Base as [dα, dβ, dγ]

        state = np.asarray(state)

        return state

    def _set_joint_effort(self, U):         # set torque U = [u1, u2, u3] to 3 joints in V-REP
        torque = [vrep.simxGetJointForce(self.clientID, self.handles['joint'][i], vrep.simx_opmode_blocking)[1]
                  for i in range(self.action_dims)]     # retrieve the current torque from the joints as [jt1, jt2, jt3]

        if len(U) != self.action_dims:
            raise Exception('the dimension of action is wrong.')

        # Give the torque and targeted velocity to joints.
        for i in range(self.action_dims):

            if U[i] > self._max_torque:         # limit the torque of each joint under max_torque
                U[i] = self._max_torque

            if np.sign(torque[i]) * np.sign(U[i]) < 0:
                self.joint_target_velocities[i] *= -1

            _ = vrep.simxSetJointTargetVelocity(self.clientID, self.handles['joint'][i],
                                                self.joint_target_velocities[i],
                                                vrep.simx_opmode_blocking)
            # Just in case
            if _ != 0:
                print('set target velocity error %s' % _)
                raise Exception('failed to set target velocity to joints.')

            _ = vrep.simxSetJointForce(self.clientID, self.handles['joint'][i],
                                       abs(U[i]), vrep.simx_opmode_blocking)
            # Just in case
            if _ != 0:
                print('set torques error %s' % _)
                raise Exception('failed to set torques to joints.')

    def grip(self):     # gripper close
        for i in range(3, 5):
            _ = vrep.simxSetJointTargetPosition(self.clientID, self.handles['joint'][i], 0, vrep.simx_opmode_blocking)
            # Just in case
            if _ != 0:
                print('set target position error %s' % _)
                raise Exception('failed to set target position to joints.')

            vrep.simxSynchronousTrigger(self.clientID)

    def _reward(self, U):       # get the current reward
        # TODO: define the immediate reward function
        # want to minimize the distance between EE_point and EE_target_position
        EE_position = vrep.simxGetObjectPosition(self.clientID, self.handles['points'][0],
                                                 -1, vrep.simx_opmode_blocking)[1]
        EE_target_position = vrep.simxGetObjectPosition(self.clientID, self.handles['points'][1],
                                                        -1, vrep.simx_opmode_blocking)[1]
        distance_square = np.sum(np.subtract(EE_position, EE_target_position)**2)

        # don't want the given torque to be too large
        U_norm = np.sum(U**2)

        reward = - self.w1 * distance_square - self.w_log * np.log(distance_square+self.alpha) - self.w2 * U_norm

        # TODO: define the function for judging whether the simulation should be stopped.
        terminal_flag = False
        if np.sqrt(distance_square) < self.tol_1 and U_norm < self.tol_2:
            terminal_flag = True

        # TODO: define the function to specify the env information.
        env_info = {}

        return reward, terminal_flag, env_info

    def reset(self):        # reset the env
        if self.num_episode != 1:
            self.terminate()
            print('Episode Ended ...')

        self._init_vrep()
        self._get_handles()
        self._configure_initial_state()
        vrep.simxStartSimulation(self.clientID, vrep.simx_opmode_blocking)
        print('_________________________ Start Episode %d _________________________' % self.num_episode)
        self.t = 1
        self.num_episode += 1
        self.cur_obs = self._get_observation()    # get current observation s_t

        return self.cur_obs

    def step(self, action):      # execute one step in env
        self._set_joint_effort(action)

        # send a synchronization trigger signal to inform V-REP to execute the next simulation step
        vrep.simxSynchronousTrigger(self.clientID)
        vrep.simxGetPingTime(self.clientID)

        Reward, ter_flag, env_info = self._reward(action)     # ter_flag: True if finish the task

        # TODO: define the scaled reward
        if self.t < self._max_episode_length * 0.8:
            scaled_reward = (self.t / self._max_episode_length) * Reward
        else:
            scaled_reward = (self.final_phase_scaler * self.t / self._max_episode_length) * Reward

        self.t += 1

        next_observation = self._get_observation()
        self.cur_obs = next_observation

        return next_observation, scaled_reward, ter_flag, env_info

    def terminate(self):        # end simulation
        vrep.simxStopSimulation(self.clientID, vrep.simx_opmode_blocking)
        vrep.simxGetPingTime(self.clientID)
        vrep.simxFinish(self.clientID)
