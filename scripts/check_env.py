from stable_baselines.common.env_checker import check_env

from MinerGymEnv import MinerGymEnv



env = MinerGymEnv(HOST=None, PORT=None)
check_env(env)