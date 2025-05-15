import gymnasium
from gymnasium.envs.registration import WrapperSpec

REGISTERED_ENVS = {
    "Carla-LaneKeep",
    "Sape-Goal0",
    "Sape-Goal1",
    "Sape-Goal2",
    "Mujoco-CartpoleSwingup",
    "Mujoco-Reacher3D",
    "Mujoco-HalfCheetah",
    "URMujoco-Reach",
    "URMujoco-PnP",
    "URRtde-Reach",
}


gymnasium.register(
    id="Carla-LaneKeep",
    entry_point="anomaly_gym.envs.carla_envs:CarlaLaneKeepEnv",
    max_episode_steps=500,
    additional_wrappers=(),
    kwargs={},
)

gymnasium.register(
    id="Sape-Goal0",
    entry_point="anomaly_gym.envs.sape_envs:SapeEnv",
    max_episode_steps=50,
    additional_wrappers=(WrapperSpec("FlattenObservation", "gymnasium.wrappers:FlattenObservation", {}),),
    kwargs={"task_id": "Goal0"},
)

gymnasium.register(
    id="Sape-Goal1",
    entry_point="anomaly_gym.envs.sape_envs:SapeEnv",
    max_episode_steps=50,
    additional_wrappers=(WrapperSpec("FlattenObservation", "gymnasium.wrappers:FlattenObservation", {}),),
    kwargs={"task_id": "Goal1"},
)

gymnasium.register(
    id="Sape-Goal2",
    entry_point="anomaly_gym.envs.sape_envs:SapeEnv",
    max_episode_steps=50,
    additional_wrappers=(WrapperSpec("FlattenObservation", "gymnasium.wrappers:FlattenObservation", {}),),
    kwargs={"task_id": "Goal2"},
)


gymnasium.register(
    id="Mujoco-CartpoleSwingup",
    entry_point="anomaly_gym.envs.mujoco_envs:MujocoCartpoleSwingupEnv",
    max_episode_steps=200,
)

gymnasium.register(
    id="Mujoco-Reacher3D",
    entry_point="anomaly_gym.envs.mujoco_envs:MujocoReacher3DEnv",
    max_episode_steps=200,
)

gymnasium.register(
    id="Mujoco-HalfCheetah",
    entry_point="anomaly_gym.envs.mujoco_envs:MujocoHalfCheetahEnv",
    max_episode_steps=200,
)


gymnasium.register(
    id="URMujoco-Reach",
    entry_point="anomaly_gym.envs.ur2l_envs:URMujocoReachEnv",
    max_episode_steps=200,
)

gymnasium.register(
    id="URRtde-Reach",
    entry_point="anomaly_gym.envs.ur2l_envs:URRtdeReachEnv",
    max_episode_steps=200,
)

gymnasium.register(
    id="URMujoco-PnP",
    entry_point="anomaly_gym.envs.ur2l_envs:URMujocoPickAndPlaceEnv",
    max_episode_steps=200,
)


# Anomaly Envs

gymnasium.register(
    id="Anom_Carla-LaneKeep",
    entry_point="anomaly_gym.envs.carla_envs:Anom_CarlaLaneKeepEnv",
    max_episode_steps=500,
    additional_wrappers=(),
    kwargs={},
)

gymnasium.register(
    id="Anom_Sape-Goal0",
    entry_point="anomaly_gym.envs.sape_envs:Anom_SapeEnv",
    max_episode_steps=50,
    additional_wrappers=(WrapperSpec("FlattenObservation", "gymnasium.wrappers:FlattenObservation", {}),),
    kwargs={"task_id": "Goal0"},
)

gymnasium.register(
    id="Anom_Sape-Goal1",
    entry_point="anomaly_gym.envs.sape_envs:Anom_SapeEnv",
    max_episode_steps=50,
    additional_wrappers=(WrapperSpec("FlattenObservation", "gymnasium.wrappers:FlattenObservation", {}),),
    kwargs={"task_id": "Goal1"},
)

gymnasium.register(
    id="Anom_Sape-Goal2",
    entry_point="anomaly_gym.envs.sape_envs:Anom_SapeEnv",
    max_episode_steps=50,
    additional_wrappers=(WrapperSpec("FlattenObservation", "gymnasium.wrappers:FlattenObservation", {}),),
    kwargs={"task_id": "Goal2"},
)


gymnasium.register(
    id="Anom_Mujoco-CartpoleSwingup",
    entry_point="anomaly_gym.envs.mujoco_envs:Anom_MujocoCartpoleSwingupEnv",
    max_episode_steps=200,
)

gymnasium.register(
    id="Anom_Mujoco-Reacher3D",
    entry_point="anomaly_gym.envs.mujoco_envs:Anom_MujocoReacher3DEnv",
    max_episode_steps=200,
)

gymnasium.register(
    id="Anom_Mujoco-HalfCheetah",
    entry_point="anomaly_gym.envs.mujoco_envs:Anom_MujocoHalfCheetahEnv",
    max_episode_steps=200,
)

gymnasium.register(
    id="Anom_URMujoco-Reach",
    entry_point="anomaly_gym.envs.ur2l_envs:Anom_URMujocoReachEnv",
    max_episode_steps=200,
)
gymnasium.register(
    id="Anom_URRtde-Reach",
    entry_point="anomaly_gym.envs.ur2l_envs:Anom_URRtdeReachEnv",
    max_episode_steps=200,
)
gymnasium.register(
    id="Anom_URMujoco-PnP",
    entry_point="anomaly_gym.envs.ur2l_envs:Anom_URMujocoPickAndPlaceEnv",
    max_episode_steps=200,
)
