"""Every setting of the environments, the observation and the reward, in one place.

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
    'd_heading':              45.0,           # action +-1 -> +-45 deg of heading
    'd_speed_kt':             20 / 3,         # action +-1 -> +-6.67 kt of CAS

    # --- decisions --------------------------------------------------------------------------
    'action_frequency':       5,              # sim seconds per decision

    # --- observation ------------------------------------------------------------------------
    'history_length':         4,              # snapshots stacked per observation
    'history_stride_s':       10,             # seconds between snapshots, a multiple of the above
    'n_rays':                 72,
    'fov_deg':                360.0,
    'lidar_range_nm':         40.0,
    'aircraft_radius_nm':     2.5,            # half the separation minimum
    'goal_width_deg':         10.0,           # std of the goal channel's bump
    'goal_distance_norm_nm':  100.0,
    'v_max_tas':              165.6,          # TAS at the CAS ceiling of 163 m/s, at 350 m

    # --- reward -----------------------------------------------------------------------------
    'w_los':                  100.0,          # per second per aircraft within 5 NM
    'w_restricted':           10.0,           # per second in a restricted area
    'w_outside':              10.0,           # per second outside the sector
    'w_drift':                1.0,            # per second, times (1 - cos drift)
    'w_time':                 0.1,            # per second
}
