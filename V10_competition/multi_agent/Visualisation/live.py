"""Play a multi-agent episode live in a pygame window, frame by frame as it is simulated.

Frames are those of visualise.py (Shared/render.py): agents in black with their goals and
lidar hits, intruder circles red while in loss of separation. Nothing is written to disk.

    python -m multi_agent.Visualisation.live --model Models/ma/mappo_adjusted/best_model.zip --seed 1

Keys: space pause/resume, right arrow one step while paused, up/down faster/slower,
r restart, esc or q quit. The window stays open on the last frame until closed.
"""

import argparse
import os

import numpy as np

from Shared.render import Renderer

# Shared/render.py selects SDL's dummy driver for offscreen frames; a window needs the real one.
if os.environ.get('SDL_VIDEODRIVER') == 'dummy':
    del os.environ['SDL_VIDEODRIVER']

import pygame

from Shared.training import load_cfg
from multi_agent.Training.train import make_env
from multi_agent.Validation.evaluate import load_policy


def episode(env, act, cfg, seed, size, lidar):
    """Yield (frame, status) per decision step, then the final frame with the episode KPIs."""
    observations, _ = env.reset(seed=seed)
    renderer = Renderer(env.scenario, cfg, size=size)
    goals = {spec.ac_id: tuple(spec.goal) for spec in env.scenario.agents}
    n_agents, step, records = len(env.agents), 0, []
    while True:
        frame = renderer.frame(env.traffic, list(env.agents), goals,
                               {a: observations[a]['own'] for a in env.agents} if lidar else None)
        status = (f'seed {seed}   t = {step * cfg["action_frequency"]:4d} s   '
                  f'flying {len(env.agents)}/{n_agents}   '
                  f'reached {sum(r["waypoint_reached"] for r in records):.0f}')
        if not env.agents:
            keys = ('waypoint_reached', 'intrusion_time', 'time_in_restricted_area', 'flight_time')
            means = {k: np.mean([r[k] for r in records]) for k in keys}
            print(f'seed {seed}: ' + ', '.join(f'{k} {v:.2f}' for k, v in means.items()))
            yield frame, status + '   (done)'
            return
        yield frame, status
        observations, _, terms, truncs, infos = env.step({a: act(observations[a]) for a in env.agents})
        records += [infos[a] for a in infos if terms[a] or truncs[a]]
        step += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=None)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--fps', type=float, default=20, help='decision steps shown per second')
    parser.add_argument('--size', type=int, default=800)
    parser.add_argument('--no-lidar', action='store_true')
    args = parser.parse_args()

    cfg = load_cfg(args.model)
    act = load_policy(args.model)
    env = make_env('2d', cfg)

    pygame.display.init()
    pygame.font.init()
    window = pygame.display.set_mode((args.size, args.size + 28))
    pygame.display.set_caption(f'MAPPO live: {args.model or "neutral policy"}')
    font = pygame.font.SysFont('consolas', 16)
    clock = pygame.time.Clock()

    frames = episode(env, act, cfg, args.seed, args.size, not args.no_lidar)
    fps, paused, finished, advance = args.fps, False, False, False
    frame, status = next(frames)
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    pygame.quit()
                    return
                if event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_RIGHT:
                    advance = True
                elif event.key == pygame.K_UP:
                    fps = min(fps * 1.5, 240)
                elif event.key == pygame.K_DOWN:
                    fps = max(fps / 1.5, 1)
                elif event.key == pygame.K_r:
                    frames = episode(env, act, cfg, args.seed, args.size, not args.no_lidar)
                    frame, status = next(frames)
                    finished = False

        if not finished and (not paused or advance):
            try:
                frame, status = next(frames)
            except StopIteration:
                finished = True
            advance = False

        window.fill((20, 20, 20))
        window.blit(pygame.surfarray.make_surface(frame.swapaxes(0, 1)), (0, 28))
        label = f'{status}   {fps:.0f} steps/s' + ('   [paused]' if paused else '')
        window.blit(font.render(label, True, (240, 240, 240)), (8, 5))
        pygame.display.flip()
        clock.tick(fps)


if __name__ == '__main__':
    main()
