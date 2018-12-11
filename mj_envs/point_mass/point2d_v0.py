import gym
import numpy as np
from gym import utils
from mj_envs import mujoco_env
from mujoco_py import MjViewer
import os


class Point2DEnvV0(mujoco_env.MujocoEnv, utils.EzPickle):

    def __init__(self, completion_bonus=0., action_scale=1., goal=None):
        self._action_scale = action_scale
        self._completion_bonus = completion_bonus
        if goal is None:
            self._goal = np.array([0., 3.])
        else:
            self._goal = goal

        curr_dir = os.path.dirname(os.path.abspath(__file__))
        mujoco_env.MujocoEnv.__init__(self, curr_dir+'/assets/point.xml', 10)

        self.action_space = gym.spaces.Box(low=-0.1, high=0.1, shape=(2, ))

    def _step(self, action):
        a = action.copy()
        a *= action * self._action_scale
        a = np.clip(a, -0.1, 0.1)
        self.do_simulation(a, self.frame_skip)

        obs = self._get_obs()

        # Calculate reward
        dist = np.linalg.norm(obs - self._goal)
        done = dist < np.linalg.norm(0.1)
        reward = -dist        

        if done:
            reward += self._completion_bonus

        return obs, reward, done, dict()

    def _get_obs(self):
        pos = self.sim.data.get_body_xpos("torso")
        return pos[:2].copy()

    def reset_model(self):
        qp = self.init_qpos.copy()
        qv = self.init_qvel.copy()
        self.set_state(qp, qv)
        self.sim.forward()
        return self._get_obs()

    def mj_viewer_setup(self):
        self.viewer = MjViewer(self.sim)
        self.sim.forward()
        self.viewer.cam.distance = 10
