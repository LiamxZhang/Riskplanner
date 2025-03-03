# 
import numpy as np
import sys
from pathlib import Path
current_file_path = Path(__file__).resolve().parent
sys.path.append(str(current_file_path.parent))

from envs.isaacgym_env import QuadrotorIsaacSim
QIS = QuadrotorIsaacSim()

from rl.PerceptDroneNav import PerceptDroneNav, DiscretePerceptDroneNav
from rl.multimodal_feature_extractor import MultiModalFeatureExtractor
from stable_baselines3 import PPO as PPO_sb3
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from rl.util.ppo import PPO

### Train ###

# Instantiate the env
vec_env = make_vec_env(DiscretePerceptDroneNav, n_envs=1, env_kwargs=dict(output_folder='results'))

# Train the agent
log_path = "./src/logs/ppo/"

# 创建回调函数
# 定期保存模型
checkpoint_callback = CheckpointCallback(
    save_freq=1000,  # 每1000步保存一次
    save_path=log_path,
    name_prefix="ppo_drone"
)

policy_kwargs=dict(
    features_extractor_class=MultiModalFeatureExtractor,
    features_extractor_kwargs=dict(cnn_output_dim=64, mlp_output_dim=64),
    net_arch=[dict(pi=[128, 128], vf=[128, 128])],  # 更大的网络架构
    log_std_init=0.0  # 初始化动作分布的标准差
)
model = PPO(
    ActorCriticPolicy, 
    vec_env,
    n_steps=64,
    batch_size=64,        # 添加batch_size
    n_epochs=4,          # 增加训练轮次
    learning_rate=3e-4,   # 设置合适的学习率
    clip_range=0.2,       # 添加clip_range
    ent_coef=0.01,        # 添加熵系数来增加探索
    policy_kwargs=policy_kwargs,
    verbose=1,
    tensorboard_log=log_path
)

# 添加回调函数列表
callbacks = [checkpoint_callback]

# 开始训练
model.learn(
    total_timesteps=1e6,
    callback=callbacks,
    tb_log_name="ppo_drone_training"  # 添加一个具体的实验名称
)
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