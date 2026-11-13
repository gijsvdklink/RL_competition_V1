"""Record a multi-agent policy as a GIF, for inspection and for the competition video.

As single_agent/Visualisation/visualise.py. All 10 agents are drawn in black with their
goals and lidar hits; an agent disappears when it captures its goal, as in the competition.

    python -m multi_agent.Visualisation.visualise --model Models/ma/<run>/best_model.zip --seeds 1 2 3 4 5
    python -m multi_agent.Visualisation.visualise --seeds 1 --no-lidar
"""

import argparse
from pathlib import Path

from Shared.harness import competition_script
from Shared.render import Renderer
from single_agent.Environment.env_bluesky import bluesky_view
from Shared.training import evaluation_cfg, load_cfg, load_policy
from multi_agent.Training.train import make_env


def record_episode(env, act, seed, cfg, lidar):
    observations, _ = env.reset(seed=seed)
    renderer = Renderer(env.scenario, cfg)
    goals = {spec.ac_id: tuple(spec.goal) for spec in env.scenario.agents}
    frames, records = [], []
    while env.agents:
        frames.append(renderer.frame(bluesky_view(), list(env.agents), goals,
                                     {a: observations[a]['own'] for a in env.agents} if lidar else None))
        observations, _, terminations, truncations, infos = env.step(
            {a: act(observations[a]) for a in env.agents})
        records += [infos[a] for a in infos if terminations[a] or truncations[a]]
    keys = ('waypoint_reached', 'flight_time', 'intrusion_time', 'time_in_restricted_area')
    print(f'seed {seed}: ' + ', '.join(f'{k} {sum(r[k] for r in records) / len(records):.1f}' for k in keys))
    return frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=None)
    parser.add_argument('--seeds', type=int, nargs='+', default=[1, 2, 3, 4, 5])
    parser.add_argument('--no-lidar', action='store_true')
    parser.add_argument('--out', default=None)
    args = parser.parse_args()

    cfg = evaluation_cfg(load_cfg(args.model))
    act = load_policy(args.model)
    env = make_env(cfg)
    frames = []
    for seed in args.seeds:
        frames += record_episode(env, act, seed, cfg, not args.no_lidar)

    run = Path(args.model).parent.name if args.model else 'neutral'
    out = args.out or f'Results/ma/{run}_seeds{"-".join(map(str, args.seeds))}.gif'
    competition_script('record_competition_gifs').save_gif(frames, out, size=512, max_frames=240)


if __name__ == '__main__':
    main()
