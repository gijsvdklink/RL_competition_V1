"""The competition's action mapping (core/actions.py), applied to the surrogate.

Action [a_hdg, a_spd] in [-1, 1] sets:
- a heading target of the current heading + a_hdg * 45 deg (HDG command);
- a CAS target of the current CAS + a_spd * 6.67 kt (SPD command).

The arithmetic, including the knots round trip, is the competition's own, so a policy
gives the same commands in both simulators. The BlueSky adapters keep the shipped
_get_action unchanged.
"""

from Shared.aero import KTS
from Shared.geo import degto180

MPS2KT = 1.94384


def apply_action(traffic, ac_id, action, cfg):
    i = traffic.id2idx(ac_id)
    new_hdg = degto180(traffic.hdg[i] + float(action[0]) * cfg['d_heading'])
    new_spd_kt = (traffic.cas[i] + float(action[1]) * cfg['d_speed_kt'] / MPS2KT) * MPS2KT
    traffic.set_heading(ac_id, new_hdg)
    traffic.set_speed(ac_id, new_spd_kt * KTS)
