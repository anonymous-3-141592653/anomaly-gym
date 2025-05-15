from classic_agents import CarlaBehaviorAgent
from gymnasium.wrappers.time_limit import TimeLimit
from tqdm import tqdm

from anomaly_gym.common.wrappers import StatisticsWrapper
from anomaly_gym.envs.carla_envs.envs.lanekeep_env import CarlaEnv

max_episode_steps = 1000

env = CarlaEnv()
env = TimeLimit(env, max_episode_steps=max_episode_steps)

print("started")
for behavior in ["cautious", "normal", "aggressive"]:
    env = StatisticsWrapper(env, keys_to_count=["collision", "lane_invasion"], keys_to_average=["ego_speed"])
    env.reset()
    agent = CarlaBehaviorAgent(env, behavior=behavior)

    for e in tqdm(range(25), position=0, desc="episode", leave=False, colour="green", ncols=80):
        obs, _ = env.reset()

        for j in tqdm(range(max_episode_steps), position=1, desc="step   ", leave=False, colour="yellow", ncols=80):
            action, _ = agent.predict(obs)
            obs, reward, term, trunc, info = env.step(action)

            if term or trunc:
                break
    print(behavior)
    print(env.total_counts)
    print(env.total_avgs)
    print("---")
print("done")
