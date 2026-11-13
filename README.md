# RL agents for the BlueSky-Gym AI4REAL-NET competition, V2

This repository holds PPO (single-agent) and MAPPO (multi-agent) policies for the
[BlueSky-Gym competition](https://github.com/TUDelft-CNS-ATM/bluesky-gym/blob/AI4REAL-NET-Competition/docs/competition/COMPETITION.md).
They follow the design of [navigation-mappo-rl](https://github.com/avbiswas/navigation-mappo-rl),
implemented in Stable-Baselines3. Everything trains and runs in the competition's own BlueSky
environments, with our observation, action and reward on their hooks.

| Part | V2 |
|---|---|
| **Observation** | 4 stacked snapshots, 10 s apart. Each holds the repo's state vector (goal direction relative to the nose) plus the commanded heading and speed: 9 values. It also holds a lidar with the repo's 3 channels (restricted areas, sector boundary, other aircraft): 120 rays over 360°, out to 92 NM. |
| **Action** | Heading = bearing to goal + a0·135°, with no turns through the tail. Speed = 100–160 m/s CAS. Action 0 is the fly-straight baseline. |
| **Decision** | every 10 s |
| **Reward** | Purely negative. −10 per new loss of separation, restricted-area entry or sector exit; −5/300 time and −12/300·(1 − cos drift) per decision (`Docs/reward_function.pdf`). |
| **Training termination** | A new violation ends the episode, as a collision does in the repo. Evaluation uses the competition's standard rules. |
| **Network** | The repo's CNN + MLP encoder, with 128-128 heads. The actor sees its own observation only. The critic adds attention over every aircraft's state (centralised training, decentralised execution). |
| **PPO** | 8 environments (as the repo), a ~1000-sample rollout as one batch, 30 epochs (target_kl 0.02), γ 0.98, lr 5e-4, VecNormalize on observations and rewards |

## Layout

| Folder | Contents |
|---|---|
| `Shared/` | config, observation and lidar, actions, reward, encoder, policy, SB3 training loop, rendering |
| `single_agent/` | the BlueSky environment, training, evaluation with the official harness, GIF recording |
| `multi_agent/` | the same for 10 agents, with one policy shared by all aircraft |
| `Docs/` | the reward function |

## Install

The competition code must sit next to this repository, on its competition branch:

```bash
git clone --branch AI4REAL-NET-Competition https://github.com/TUDelft-CNS-ATM/bluesky-gym.git
git clone --branch v2 https://github.com/gijsvdklink/RL_competition_V1.git RL_COMPETITION_V2
cd RL_COMPETITION_V2
UV_PROJECT_ENVIRONMENT=~/.venvs/competition-rl uv sync
source ~/.venvs/competition-rl/bin/activate
```

The venv is kept outside the project on purpose. macOS hides `.pth` files inside an
iCloud-synced Documents folder, and the editable install of bluesky-gym then fails to import.

## Train

Every terminal command, in order, is also in `tutorial.txt`.

```bash
python -m single_agent.Training.train --steps 1e6 --name ppo
python -m multi_agent.Training.train --steps 5e6 --name mappo
python -m tensorboard.main --logdir Models --port 6007
```

TensorBoard shows three groups:
- **`rollout/*`:** the training episodes. The mean and `_std` of the reward and every competition KPI, the share of episodes ended by a violation, and the number of episodes.
- **`eval/*`:** every 100k steps, 20 deterministic episodes on held-out scenarios, under the standard rules.
- **`train/*`:** PPO's own diagnostics.

## Evaluate and record

```bash
python -m single_agent.Validation.evaluate --model Models/sa/ppo/best_model.zip          # official harness, seed 42
python -m single_agent.Visualisation.visualise --model Models/sa/ppo/best_model.zip --seeds 1 2 3 4 5
python -m single_agent.Visualisation.visualise --seeds 42                                # the fly-straight baseline
```

For the multi-agent model, use the `multi_agent.` modules and a model under `Models/ma/`. A
model's observation statistics (`*_vecnormalize.pkl`) are saved beside it and applied
automatically.
