"""What both trainers share: the PPO settings, the command line, and the TensorBoard KPIs.

Both train with Stable-Baselines3 PPO, in BlueSky by default.
VecNormalize scales the reward. The V2 adjusted schedule uses 100 decisions per
rollout slot by default, with 30 epochs and the whole rollout buffer as one batch.

--algo sac trains SB3's SAC with its default settings instead (SAC_SETTINGS). Its actor and
its two Q-critics each get their own copy of the observation encoder (Shared/encoder.py) on
the aircraft's own observation; the joint tokens of the centralised critic are dropped before
the replay buffer, which only has to hold 'own'. Multi-agent SAC is parameter sharing, as
MAPPO: every aircraft is one sub-env and adds its transitions to the shared buffer.

TensorBoard, under Models/<sa|ma>/<run>/:

    rollout/ep_rew_mean  the total reward per (agent-)episode, SB3's own, over the last 100
    rollout/<kpi>        the goal completion rate and the nine scored metrics, averaged over
                         the training episodes that ended in that rollout
    eval/*               every eval_every steps, eval_episodes deterministic episodes on
                         held-out scenarios (seed 7, never the competition's 42): the goal
                         completion rate, the nine metrics and, for multi-agent, the
                         team's total reward

The model with the best eval/total_reward is kept as best_model.zip.
"""

import argparse
import json
import signal
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import VecEnvWrapper, VecNormalize

from Shared.encoder import ObservationEncoder

from Shared.config import CONFIG
from Shared.metrics import METRIC_KEYS

SETTINGS = {
    'learning_rate':   3e-4,
    'n_steps':         100,       # decisions per aircraft slot per rollout
    'batch_size':      'full',
    'n_epochs':        30,
    'gamma':           0.99,
    'gae_lambda':      0.95,
    'clip_range':      0.2,
    'ent_coef':        0.0,
    'vf_coef':         0.5,
    'max_grad_norm':   0.5,
}

# SB3's SAC defaults (stable_baselines3/sac/sac.py), written out so config.json records them
SAC_SETTINGS = {
    'learning_rate':   3e-4,
    'buffer_size':     1_000_000,  # transitions, shared by all envs
    'learning_starts': 100,
    'batch_size':      256,
    'tau':             0.005,
    'gamma':           0.99,
    'train_freq':      1,          # one vec-env step ...
    'gradient_steps':  1,          # ... then one gradient step
    'ent_coef':        'auto',
}

SAVE_EVERY      = 20_000           # aircraft steps
EVAL_EVERY      = 100_000          # steps
EVAL_EPISODES   = 20
VALIDATION_SEED = 7


def parse_args(doc, default_envs):
    parser = argparse.ArgumentParser(description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--env', choices=['2d', 'bluesky'], default='bluesky')
    parser.add_argument('--algo', choices=['ppo', 'sac'], default='ppo')
    parser.add_argument('--buffer-size', type=int, default=SAC_SETTINGS['buffer_size'],
                        help='SAC replay buffer; ~7 GB of RAM per million transitions')
    parser.add_argument('--decision-step', type=int, default=CONFIG['action_frequency'], help='seconds')
    parser.add_argument('--history-stride', type=int, default=CONFIG['history_stride_s'], help='seconds')
    parser.add_argument('--steps', type=float, help='aircraft-step limit; omit to train until stopped')
    parser.add_argument('--save-every', type=int, default=SAVE_EVERY,
                        help='aircraft steps between checkpoints')
    parser.add_argument('--n-envs', type=int, default=default_envs)
    parser.add_argument('--n-steps', type=int, default=SETTINGS['n_steps'], help='rollout per env')
    parser.add_argument('--n-epochs', type=int, default=SETTINGS['n_epochs'])
    parser.add_argument('--batch-size', default=SETTINGS['batch_size'],
                        help="an int, or 'full' for the whole rollout buffer")
    parser.add_argument('--net-width', type=int, default=128, help='neurons per layer of the 2-layer heads')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--name', default=None)
    parser.add_argument('--resume', default=None, help='a .zip; its vecnormalize.pkl is read beside it')
    args = parser.parse_args()
    if args.save_every <= 0:
        parser.error('--save-every must be positive')
    return args


def make_training_env(make_env, env_kind, cfg):
    # The trainer handles Ctrl+C and closes its workers after saving.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    return make_env(env_kind, cfg)


def run_config(args):
    return {**CONFIG, 'action_frequency': args.decision_step, 'history_stride_s': args.history_stride}


def load_cfg(model_path):
    """The cfg a saved model was trained with, from config.json in its run folder; the
    checkpoints/ folder sits one level deeper. CONFIG for the neutral policy."""
    if model_path is None:
        return CONFIG
    for folder in Path(model_path).parents[:2]:
        if (folder / 'config.json').exists():
            return {**CONFIG, **json.loads((folder / 'config.json').read_text())['cfg']}
    raise FileNotFoundError(f'no config.json beside {model_path}')


def train(kind, args, cfg, venv, policy, policy_kwargs, evaluate):
    """Train one run: kind is 'sa' or 'ma', venv the un-normalised VecEnv, evaluate(model, n)
    returns a dict of KPIs over n validation episodes."""
    torch.set_num_threads(2)
    if args.algo == 'sac':
        return train_sac(kind, args, cfg, venv, evaluate)
    full = args.batch_size == 'full'
    settings = {**SETTINGS, 'n_steps': args.n_steps, 'n_epochs': args.n_epochs,
                'batch_size': args.n_steps * venv.num_envs if full else int(args.batch_size)}
    run_dir = Path('Models') / kind / (args.name or f'{args.env}_seed{args.seed}')
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / 'config.json').write_text(json.dumps(
        {'cfg': cfg, 'settings': settings, 'n_envs': args.n_envs, 'env': args.env}, indent=2))

    if args.resume:
        venv = VecNormalize.load(str(Path(args.resume).parent / 'vecnormalize.pkl'), venv)
        model = PPO.load(args.resume, venv, tensorboard_log=str(run_dir))
    else:
        venv = VecNormalize(venv, norm_obs=False, norm_reward=True, gamma=settings['gamma'])
        model = PPO(policy, venv, policy_kwargs=policy_kwargs, tensorboard_log=str(run_dir),
                    seed=args.seed, verbose=1, **settings)

    callbacks = [CheckpointCallback(max(1, args.save_every // venv.num_envs), str(run_dir / 'checkpoints'),
                                    save_vecnormalize=True),
                 TrainingKPIs(),
                 EvaluationKPIs(evaluate, run_dir)]
    learn_and_save(model, venv, run_dir, args.steps, callbacks,
                   reset_num_timesteps=not args.resume)


class OwnObservation(VecEnvWrapper):
    """Hands SAC only the actor's 'own' observation; the critic's joint tokens are dropped."""

    def __init__(self, venv):
        super().__init__(venv, observation_space=venv.observation_space['own'])

    def reset(self):
        return self.venv.reset()['own']

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        for info in infos:
            if isinstance(info.get('terminal_observation'), dict):
                info['terminal_observation'] = info['terminal_observation']['own']
        return obs['own'], rewards, dones, infos


class OwnFeatures(BaseFeaturesExtractor):
    """The observation encoder on the flat 'own' observation, for SAC's actor and critics."""

    def __init__(self, observation_space, n_rays, history_length, features_dim=256):
        super().__init__(observation_space, features_dim)
        self.encoder = ObservationEncoder(n_rays, history_length, features_dim)

    def forward(self, observations):
        return self.encoder(observations)


def train_sac(kind, args, cfg, venv, evaluate):
    settings = {**SAC_SETTINGS, 'buffer_size': args.buffer_size}
    run_dir = Path('Models') / kind / (args.name or f'sac_{args.env}_seed{args.seed}')
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / 'config.json').write_text(json.dumps(
        {'cfg': cfg, 'algo': 'sac', 'settings': settings, 'n_envs': args.n_envs, 'env': args.env},
        indent=2))

    venv = VecNormalize(OwnObservation(venv), norm_obs=False, norm_reward=True, gamma=settings['gamma'])
    policy_kwargs = dict(features_extractor_class=OwnFeatures,
                         features_extractor_kwargs=dict(n_rays=cfg['n_rays'],
                                                        history_length=cfg['history_length']))
    model = SAC('MlpPolicy', venv, policy_kwargs=policy_kwargs, tensorboard_log=str(run_dir),
                seed=args.seed, verbose=1, **settings)
    callbacks = [CheckpointCallback(max(1, args.save_every // venv.num_envs), str(run_dir / 'checkpoints'),
                                    save_vecnormalize=True),
                 TrainingKPIs(),
                 EvaluationKPIs(evaluate, run_dir)]
    learn_and_save(model, venv, run_dir, args.steps, callbacks)


class StopOnInterrupt(BaseCallback):
    def __init__(self):
        super().__init__()
        self.requested = False

    def request_stop(self, signum, frame):
        self.requested = True

    def _on_step(self):
        return not self.requested


def learn_and_save(model, venv, run_dir, steps, callbacks, reset_num_timesteps=True):
    stop = StopOnInterrupt()
    previous_handler = signal.signal(signal.SIGINT, stop.request_stop)
    try:
        total_timesteps = float('inf') if steps is None else int(steps)
        model.learn(total_timesteps, callback=[stop, *callbacks],
                    reset_num_timesteps=reset_num_timesteps)
        model.save(run_dir / 'final_model')
        venv.save(str(run_dir / 'vecnormalize.pkl'))
        if stop.requested:
            print(f'Stopped at {model.num_timesteps:,} steps. Saved {run_dir / "final_model.zip"}',
                  flush=True)
    finally:
        try:
            venv.close()
        finally:
            signal.signal(signal.SIGINT, previous_handler)


def policy_input(model, obs):
    """obs as the model takes it: the full dict for PPO, only 'own' for SAC."""
    return obs if hasattr(model.observation_space, 'spaces') else obs['own']


def load_model(path):
    """PPO or SAC, from the algo recorded in config.json beside the model."""
    for folder in Path(path).parents[:2]:
        if (folder / 'config.json').exists():
            algo = json.loads((folder / 'config.json').read_text()).get('algo', 'ppo')
            return (SAC if algo == 'sac' else PPO).load(path, device='cpu')
    return PPO.load(path, device='cpu')


def kpi_means(records):
    kpis = {key: float(np.mean([r[key] for r in records])) for key in METRIC_KEYS}
    kpis['goal_completion_rate_pct'] = 100.0 * kpis['waypoint_reached']
    return kpis


class TrainingKPIs(BaseCallback):
    """The KPIs of every training episode as it ends, averaged per rollout."""

    def _on_step(self):
        for info in self.locals['infos']:
            if 'episode' in info and 'waypoint_reached' in info:
                for key, value in kpi_means([info]).items():
                    self.logger.record_mean(f'rollout/{key}', value)
        return True


class EvaluationKPIs(BaseCallback):
    """Deterministic episodes on held-out scenarios every EVAL_EVERY steps."""

    def __init__(self, evaluate, run_dir):
        super().__init__()
        self.evaluate, self.run_dir = evaluate, run_dir
        self.next_eval, self.best = EVAL_EVERY, -np.inf

    def _on_step(self):
        if self.num_timesteps < self.next_eval:
            return True
        self.next_eval += EVAL_EVERY
        kpis = self.evaluate(self.model, EVAL_EPISODES)
        for key, value in kpis.items():
            self.logger.record(f'eval/{key}', value)
        if kpis['total_reward'] > self.best:
            self.best = kpis['total_reward']
            self.model.save(self.run_dir / 'best_model')
        return True
