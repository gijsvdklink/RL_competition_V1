"""V8 reward applied to increments in the competition metrics.

Single-agent uses ownship separation metrics. Multi-agent uses the sum of separation
increments across the aircraft that started this decision, including those that arrive
during it. Each aircraft pays its own airspace and flight-time penalties.
"""

METRICS = ('flight_time', 'intrusion_time', 'intrusion_events', 'time_in_restricted_area',
           'restricted_area_events', 'time_outside_sector', 'sector_exit_events')


def gained(now, before):
    return {key: now[key] - before[key] for key in METRICS}


def reward(weights, own, separation=None):
    if separation is None:
        separation = own
    return (weights['los_s'] * separation['intrusion_time']
            + weights['los_event'] * separation['intrusion_events']
            + weights['restricted_s'] * own['time_in_restricted_area']
            + weights['restricted_event'] * own['restricted_area_events']
            + weights['outside_s'] * own['time_outside_sector']
            + weights['outside_event'] * own['sector_exit_events']
            + weights['time_s'] * own['flight_time'])


def team_reward(weights, metrics, before, agent, flying=None):
    if flying is None:
        flying = before
    own = gained(metrics[agent], before[agent])
    separation = {
        key: sum(metrics[ac][key] - before[ac][key] for ac in flying)
        for key in ('intrusion_time', 'intrusion_events')
    }
    return reward(weights, own, separation)
