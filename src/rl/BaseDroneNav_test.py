# 
import numpy as np
import sys
from pathlib import Path
current_file_path = Path(__file__).resolve().parent
sys.path.append(str(current_file_path.parent))

from envs.isaacgym_env import QuadrotorIsaacSim
QIS = QuadrotorIsaacSim()

from rl.BaseDroneNav import BaseDroneNav
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.env_util import make_vec_env

### Test ###
env = BaseDroneNav()
# check_env(env, warn=True)

print(env.observation_space)
print(env.action_space)
print(env.action_space.sample())

step = 0
while QIS.is_running():
    step += 1
    print(f"Step {step}")
    action = np.random.uniform(low=-1.0, high=1.0, size=(4,)).astype(np.float32)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    print("obs=", obs, "reward=", reward, "done=", done)
    QIS.update()
    if done:
        print("Goal reached!", "reward=", reward)
        break


# Terminate the APP
QIS.stop()