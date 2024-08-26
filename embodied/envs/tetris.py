from gymnasium.wrappers import ResizeObservation
from tetris_gymnasium.wrappers.observation import RgbObservation

import gymnasium as gym

from embodied.envs.from_gym import FromGym


class Tetris(FromGym):

  def __init__(self, task, size=(96, 96), **kwargs):
    env = gym.make("tetris_gymnasium/Tetris", render_mode="rgb_array")
    env = RgbObservation(env)
    env = ResizeObservation(env, size)

    super().__init__(env, **kwargs)
