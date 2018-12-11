from gym.envs.registration import register
from mj_envs.mujoco_env import MujocoEnv

# Simple point environment
register(
	id='point2d-v0',
	entry_point='mj_envs.point_mass:Point2DEnvV0',
	max_episode_steps=200,
)
from mj_envs.point_mass.point2d_v0 import Point2DEnvV0
