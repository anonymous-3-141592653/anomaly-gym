import numpy as np

EXPERT_REWARDS = {
    "Carla-LaneKeep": -47.263,
    "Mujoco-CartpoleSwingup": 182.664,
    "Mujoco-HalfCheetah": 2618.671,
    "Mujoco-Reacher3D": -55.806,
    "Sape-Goal0": -2.923,
    "Sape-Goal1": -6.328,
    "Sape-Goal2": -9.338,
    "UR3-PickAndDrop2D": -8.867,
    "UR3-PickAndPlace3D": 0.286,
    "UR3-Reach2D": -0.708,
    "URMujoco-Reach": -143.98,
    "URRtde-Reach": -150.976,
    "URMujoco-PnP": -4.931494,
}


RANDOM_REWARDS = {
    "Carla-LaneKeep": -220.78,
    "Mujoco-CartpoleSwingup": -4.111,
    "Mujoco-HalfCheetah": -56.528,
    "Mujoco-Reacher3D": -224.855,
    "Sape-Goal0": -122.551,
    "Sape-Goal1": -151.246,
    "Sape-Goal2": -177.359,
    "UR3-PickAndDrop2D": -50.56,
    "UR3-PickAndPlace3D": -36.854,
    "UR3-Reach2D": -34.462,
    "URMujoco-Reach": -248.341,
    "URRtde-Reach": -244.91,
    "URMujoco-PnP": -71.087654,
}

EXPERT_EP_LENGTHS = {
    "Carla-LaneKeep": 500,
    "Mujoco-CartpoleSwingup": 200,
    "Mujoco-HalfCheetah": 200,
    "Mujoco-Reacher3D": 200,
    "Sape-Goal0": 13,
    "Sape-Goal1": 14,
    "Sape-Goal2": 15,
    "UR3-PickAndDrop2D": 34,
    "UR3-PickAndPlace3D": 24,
    "UR3-Reach2D": 23,
    "URMujoco-Reach": 200,
    "URRtde-Reach": 200,
    "URMujoco-PnP": 200,
}


def compute_normalized_score(reward_avg, expert_score, random_score):
    return (reward_avg - random_score) / (expert_score - random_score)


def get_normalized_score(env_id, reward_avg):
    expert_score = EXPERT_REWARDS.get(env_id, np.nan)
    random_score = RANDOM_REWARDS.get(env_id, np.nan)
    return compute_normalized_score(reward_avg, expert_score, random_score)
