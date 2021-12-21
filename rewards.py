import numpy as np

last_measure = []

def delaydiff_pressure09(signals):
    r1 = pressure(signals)
    rewards = dict()
    i = 0
    nums = len(signals)
    global last_measure
    if last_measure == []:
        last_measure = [0 for i in range(nums)]
    for signal in signals:
        total = 0
        for lane in signals[signal].lanes:
            total += signals[signal].full_observation[lane]['total_wait']
        rewards[signal] = (last_measure[i] - total) * 0.9 + r1[signal] * 0.1
        last_measure[i] = total
        i = i + 1
    return rewards

def delaydiff_pressure05(signals):
    r1 = pressure(signals)
    rewards = dict()
    i = 0
    nums = len(signals)
    global last_measure
    if last_measure == []:
        last_measure = [0 for i in range(nums)]
    for signal in signals:
        total = 0
        for lane in signals[signal].lanes:
            total += signals[signal].full_observation[lane]['total_wait']
        rewards[signal] = (last_measure[i] - total) * 0.5 + r1[signal] * 0.5
        last_measure[i] = total
        i = i + 1
    return rewards

# add by longshui
def delaydiff_pressure03(signals):
    r1 = pressure(signals)
    rewards = dict()
    i = 0
    nums = len(signals)
    global last_measure
    if last_measure == []:
        last_measure = [0 for i in range(nums)]
    for signal in signals:
        total = 0
        for lane in signals[signal].lanes:
            total += signals[signal].full_observation[lane]['total_wait']
        rewards[signal] = (last_measure[i] - total) * 0.3 + r1[signal] * 0.7
        last_measure[i] = total
        i = i + 1
    return rewards

def delaydiff_pressure0(signals):
    rewards = dict()
    i = 0
    nums = len(signals)
    global last_measure
    if last_measure == []:
        last_measure = [0 for i in range(nums)]
    for signal in signals:
        total = 0
        for lane in signals[signal].lanes:
            total += signals[signal].full_observation[lane]['total_wait']
        rewards[signal] = (last_measure[i] - total)
        last_measure[i] = total
        i = i + 1
    return rewards

# add by longshui
def delaydiff_pressure01(signals):
    r1 = pressure(signals)
    rewards = dict()
    i = 0
    nums = len(signals)
    global last_measure
    if last_measure == []:
        last_measure = [0 for i in range(nums)]
    for signal in signals:
        total = 0
        for lane in signals[signal].lanes:
            total += signals[signal].full_observation[lane]['total_wait']
        rewards[signal] = (last_measure[i] - total) * 0.1 + r1[signal] * 0.9
        last_measure[i] = total
        i = i + 1
    return rewards

# add by longshui
def delaydiff_pressure(signals):
    r1 = pressure(signals)
    rewards = dict()
    i = 0
    nums = len(signals)
    global last_measure
    if last_measure == []:
        last_measure = [0 for i in range(nums)]
    for signal in signals:
        total = 0
        for lane in signals[signal].lanes:
            total += signals[signal].full_observation[lane]['total_wait']
        rewards[signal] = (last_measure[i] - total) * 0.7 + r1[signal] * 0.3
        last_measure[i] = total
        i = i + 1
    return rewards



def wait(signals):
    rewards = dict()
    for signal_id in signals:
        total_wait = 0
        for lane in signals[signal_id].lanes:
            total_wait += signals[signal_id].full_observation[lane]['total_wait']

        rewards[signal_id] = -total_wait
    return rewards


def wait_norm(signals):
    rewards = dict()
    for signal_id in signals:
        total_wait = 0
        for lane in signals[signal_id].lanes:
            total_wait += signals[signal_id].full_observation[lane]['total_wait']

        rewards[signal_id] = np.clip(-total_wait/224, -4, 4).astype(np.float32)
    return rewards


def pressure(signals):
    rewards = dict()
    for signal_id in signals:
        queue_length = 0
        for lane in signals[signal_id].lanes:
            queue_length += signals[signal_id].full_observation[lane]['queue']

        for lane in signals[signal_id].outbound_lanes:
            dwn_signal = signals[signal_id].out_lane_to_signalid[lane]
            if dwn_signal in signals[signal_id].signals:
                queue_length -= signals[signal_id].signals[dwn_signal].full_observation[lane]['queue']

        rewards[signal_id] = -queue_length
    return rewards
