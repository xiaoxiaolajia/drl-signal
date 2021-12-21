import rewards
import states

from agents.stochastic import STOCHASTIC
from agents.maxwave import MAXWAVE
from agents.maxpressure import MAXPRESSURE
from agents.pfrl_dqn import IDQN

agent_configs = {
    # *VAL configs have distance settings according to the validation scenarios
    'MAXWAVEVAL': {
        'agent': MAXWAVE,
        'state': states.wave,
        'reward': rewards.wait,
        'max_distance': 50
    },
    'MAXPRESSUREVAL': {
        'agent': MAXPRESSURE,
        'state': states.mplight,
        'reward': rewards.wait,
        'max_distance': 9999
    },
    'STOCHASTIC': {
        'agent': STOCHASTIC,
        'state': states.mplight,
        'reward': rewards.wait,
        'max_distance': 1
    },
    'MAXWAVE': {
        'agent': MAXWAVE,
        'state': states.wave,
        'reward': rewards.wait,
        'max_distance': 50
    },
    'MAXPRESSURE': {
        'agent': MAXPRESSURE,
        'state': states.mplight,
        'reward': rewards.wait,
        'max_distance': 200
    },
    'IDQN': {
        'agent': IDQN,
        'state': states.drqlong,
        'reward': rewards.delaydiff_pressure0,
        'max_distance': 200,
        'BATCH_SIZE': 32,
        'GAMMA': 0.8,
        'EPS_START': 1.0,
        'EPS_END': 0.0,
        'EPS_DECAY': 220,
        'TARGET_UPDATE': 500
    },


}
