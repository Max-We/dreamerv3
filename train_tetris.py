from datetime import datetime

import gymnasium as gym
from gymnasium.wrappers import ResizeObservation

from tetris_gymnasium.envs.tetris import Tetris
from tetris_gymnasium.wrappers.observation import RgbObservation


def main():

  import warnings
  import dreamerv3
  from dreamerv3 import embodied
  warnings.filterwarnings('ignore', '.*truncated to dtype int32.*')

  # See configs.yaml for all options.
  config = embodied.Config(dreamerv3.configs['defaults'])
  config = config.update(dreamerv3.configs['large'])
  config = config.update(dreamerv3.configs['atari'])
  config = config.update({
      'logdir': f'~/logdir/run_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
      # 'run.log_every': 30,  # Seconds
      # 'jax.prealloc': False,
      # 'encoder.mlp_keys': '$^',
      # 'decoder.mlp_keys': '$^',
      # 'encoder.cnn_keys': 'image',
      # 'decoder.cnn_keys': 'image',
      'jax.platform': 'cpu',
  })
  config = embodied.Flags(config).parse()

  logdir = embodied.Path(config.logdir)
  step = embodied.Counter()
  logger = embodied.Logger(step, [
      embodied.logger.TerminalOutput(),
      embodied.logger.JSONLOutput(logdir, 'metrics.jsonl'),
      embodied.logger.TensorBoardOutput(logdir),
      embodied.logger.WandBOutput(r'.*', logdir, config),
      # embodied.logger.MLFlowOutput(logdir.name),
  ])

  import crafter
  from embodied.envs import from_gym
  # env = crafter.Env(
  #     area=(96, 96), view=(9, 9), size=(96, 96)
  # )  # Replace this with your Gym env.
  # env = ResizeObservation(env, (64, 64))

  env = gym.make("tetris_gymnasium/Tetris", render_mode="rgb_array")
  env = RgbObservation(env)
  env = ResizeObservation(env, (64, 64))

  # env = gym.make("CarRacing-v2", render_mode="human")
  # env = ResizeObservation(env, (64, 64))

  env = from_gym.FromGym(env, obs_key='image')  # Or obs_key='vector'.
  env = dreamerv3.wrap_env(env, config)
  env = embodied.BatchEnv([env], parallel=True)

  agent = dreamerv3.Agent(env.obs_space, env.act_space, step, config)
  replay = embodied.replay.Uniform(
      config.batch_length, config.replay_size, logdir / 'replay')
  args = embodied.Config(
      **config.run, logdir=config.logdir,
      batch_steps=config.batch_size * config.batch_length)
  embodied.run.train(agent, env, replay, logger, args)
  # embodied.run.eval_only(agent, env, logger, args)


if __name__ == '__main__':
  main()
