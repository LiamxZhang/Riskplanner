# 
import numpy as np
import sys
from pathlib import Path
current_file_path = Path(__file__).resolve().parent
sys.path.append(str(current_file_path.parent))

from envs.isaacgym_env import QuadrotorIsaacSim
QIS = QuadrotorIsaacSim()

from rl.PerceptDroneNav import PerceptDroneNav
from rl.multimodal_feature_extractor import MultiModalFeatureExtractor
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.env_util import make_vec_env

### Train ###

# Instantiate the env
vec_env = make_vec_env(PerceptDroneNav, n_envs=1, env_kwargs=dict(output_folder='results'))

# Train the agent
log_path = "./src/logs/ppo/"
policy_kwargs = dict(
    features_extractor_class=MultiModalFeatureExtractor,
    features_extractor_kwargs=dict(cnn_output_dim=256, mlp_output_dim=64)
)
model = PPO(ActorCriticPolicy, vec_env, policy_kwargs=policy_kwargs, verbose=1, tensorboard_log=log_path)
model.learn(1e6)
model.save(log_path + "ppo_model")  # save to ppo_model.zip

### Test ###

model = PPO.load(log_path + "ppo_model", print_system_info=True)
# using the vecenv
obs = vec_env.reset()
n_steps = 20
for step in range(n_steps):
    action, _ = model.predict(obs, deterministic=True)
    print(f"Step {step + 1}")
    print("Action: ", action)
    obs, reward, done, info = vec_env.step(action)
    print("obs=", obs, "reward=", reward, "done=", done)
    vec_env.render()
    if done:
        # Note that the VecEnv resets automatically
        # when a done signal is encountered
        print("Goal reached!", "reward=", reward)
        break

# Terminate the APP
QIS.stop()