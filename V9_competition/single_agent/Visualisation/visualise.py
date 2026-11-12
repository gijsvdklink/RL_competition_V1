"""Record a single-agent policy as a GIF, for inspection and for the competition video.

Each seed is one episode. All seeds go into one GIF, written by the competition's own
save_gif (scripts/record_competition_gifs.py). Frames come from Shared/render.py: the scene
as the competition draws it, plus the agent's lidar hits.

    python -m single_agent.Visualisation.visualise --model Models/sa/<run>/best_model.zip --seeds 1 2 3 4 5
    python -m single_agent.Visualisation.visualise --seeds 3 --env 2d     # neutral policy, surrogate
"""

import argparse
from pathlib import Path

from Shared.harness import competition_script
from Shared.render import Renderer
from Shared.training import load_cfg
from single_agent.Training.train import make_env
from single_agent.Validation.evaluate import load_policy


def traffic_of(env, env_kind):
    if env_kind == '2d':
        return env.traffic
    from single_agent.Environment.env_bluesky import bluesky_view
    return bluesky_view()


def record_episode(env, env_kind, act, seed, cfg):
    obs, _ = env.reset(seed=seed)
    renderer = Renderer(env.scenario, cfg)
    goal = {env.agent: tuple(env.scenario.agents[0].goal)}
    frames, done, truncated, info = [], False, False, {}
    while not (done or truncated):
        frames.append(renderer.frame(traffic_of(env, env_kind), [env.agent], goal, {env.agent: obs['own']}))
        obs, _, done, truncated, info = env.step(act(obs))
    print(f'seed {seed}: ' + ', '.join(f'{k} {v:g}' for k, v in info.items()))
    return frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=None)
    parser.add_argument('--env', choices=['2d', 'bluesky'], default='bluesky')
    parser.add_argument('--seeds', type=int, nargs='+', default=[1, 2, 3, 4, 5])
    parser.add_argument('--out', default=None)
    parser.add_argument('--mp4', action='store_true', help='write an MP4 with every step instead of a GIF')
    parser.add_argument('--fps', type=int, default=20)
    args = parser.parse_args()

    cfg = load_cfg(args.model)
    act = load_policy(args.model)
    env = make_env(args.env, cfg)
    frames = []
    for seed in args.seeds:
        frames += record_episode(env, args.env, act, seed, cfg)

    run = Path(args.model).parent.name if args.model else 'neutral'
    out = args.out or f'Results/sa/{run}_{args.env}_seeds{"-".join(map(str, args.seeds))}'
    if args.mp4:
        from Shared.video import save_mp4
        save_mp4(frames, out if out.endswith('.mp4') else out + '.mp4', fps=args.fps)
        return
    out = out if out.endswith('.gif') else out + '.gif'
    competition_script('record_competition_gifs').save_gif(frames, out, size=512, max_frames=240)


if __name__ == '__main__':
    main()
