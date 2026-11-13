"""Every setting of the environments, the observation, the action and the reward, in one place.

The first block repeats the competition's fixed parameters (the constructor defaults of
CompetitionEnv / CompetitionZooEnv); they are scored on these values and must not change.
Everything below it is ours to tune. Units: NM, m/s, degrees and seconds unless a key
names its own unit.
"""

CONFIG = {
    # --- competition, fixed ---------------------------------------------------------------
    'n_intruders':            10,
    'n_agents':               10,
    'n_obstacles':            5,
    'episode_time_limit_s':   3000.0,
    'intrusion_distance_nm':  5.0,
    'distance_margin_km':     5.0,
    'ac_spd':                 150.0,          # CAS at creation
    'altitude_m':             350.0,          # acalt=350 is read by BlueSky as metres
    'center':                 (52.0, 4.0),

    # --- decisions and actions --------------------------------------------------------------
    'action_frequency':       10,             # sim seconds per decision: 300 per episode
    'max_offset_deg':         135.0,          # a0 = +-1 -> heading = goal bearing +-135 deg
    'max_turn_deg':           170.0,          # never command a turn through the tail
    'cas_nominal':            150.0,          # a1 = 0 -> the start speed
    'cas_up':                 10.0,           # a1 = +1 -> 160 m/s, just under the 163 ceiling
    'cas_down':               50.0,           # a1 = -1 -> 100 m/s

    # --- observation ------------------------------------------------------------------------
    'history_length':         4,              # snapshots stacked per observation
    'history_stride_s':       10,             # consecutive decisions: motion over 30 s
    'n_rays':                 120,            # 3 deg apart: a 5 NM disc is hit out to full range
    'fov_deg':                360.0,
    'lidar_range_nm':         92.0,           # 170 km, the longest start-goal distance
    'aircraft_radius_nm':     5.0,            # the separation: a hit reads distance to a LoS
    'goal_distance_norm_nm':  100.0,
    'v_max_tas':              165.6,          # TAS at the CAS ceiling of 163 m/s, at 350 m

    # --- reward and training termination ------------------------------------------------------
    'w_los':                  10.0,           # once per new loss of separation
    'w_restricted':           10.0,           # once per new restricted-area entry
    'w_outside':              10.0,           # once per new sector exit
    'w_time_episode':         5.0,            # time penalty over a full 3000 s episode
    'w_drift_episode':        12.0,           # a full episode at 90 deg off: 0.04 per decision
    'terminate_on_violation': True,           # training only; evaluation keeps the official rules
}
