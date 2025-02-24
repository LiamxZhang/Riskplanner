import warnings
from collections import OrderedDict
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Sequence, Type

import gymnasium as gym
import numpy as np

from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvIndices, VecEnvObs, VecEnvStepReturn
from stable_baselines3.common.vec_env.patch_gym import _patch_env
from stable_baselines3.common.vec_env.util import copy_obs_dict, dict_to_obs, obs_space_info

from envs.isaacgym_env import QuadrotorIsaacSim


class IsaacDummyVecEnv(VecEnv):
    """
    Creates a simple vectorized wrapper for multiple environments, calling each environment in sequence on the current
    Python process. This is useful for computationally simple environment such as ``Cartpole-v1``,
    as the overhead of multiprocess or multithread outweighs the environment computation time.
    This can also be used for RL methods that
    require a vectorized environment, but that you want a single environments to train with.

    :param env_fns: a list of functions
        that return environments to vectorize
    :raises ValueError: If the same environment instance is passed as the output of two or more different env_fn.
    """

    actions: np.ndarray

    def __init__(self, env_fns: List[Callable[[], gym.Env]]):
        self.envs = [_patch_env(fn()) for fn in env_fns]
        if len(set([id(env.unwrapped) for env in self.envs])) != len(self.envs):
            raise ValueError(
                "You tried to create multiple environments, but the function to create them returned the same instance "
                "instead of creating different objects. "
                "You are probably using `make_vec_env(lambda: env)` or `DummyVecEnv([lambda: env] * n_envs)`. "
                "You should replace `lambda: env` by a `make_env` function that "
                "creates a new instance of the environment at every call "
                "(using `gym.make()` for instance). You can take a look at the documentation for an example. "
                "Please read https://github.com/DLR-RM/stable-baselines3/issues/1151 for more information."
            )
        env = self.envs[0]
        super().__init__(len(env_fns), env.observation_space, env.action_space)
        obs_space = env.observation_space
        self.keys, shapes, dtypes = obs_space_info(obs_space)

        self.buf_obs = OrderedDict([(k, np.zeros((self.num_envs, *tuple(shapes[k])), dtype=dtypes[k])) for k in self.keys])
        self.buf_dones = np.zeros((self.num_envs,), dtype=bool)
        self.buf_rews = np.zeros((self.num_envs,), dtype=np.float32)
        self.buf_infos: List[Dict[str, Any]] = [{} for _ in range(self.num_envs)]
        self.metadata = env.metadata
        self._async_start_time = 0.0
        self._async_duration = 0.0
        self._step1_time = 0.0
        self._update_time = 0.0
        self._stats_interval = 20  # 统计间隔
        self._step_count = 0
        self._time_accumulator = {
            'async_total': 0.0,
            'step1_total': 0.0,
            'update_total': 0.0,
            'wait_total': 0.0,
            'step_total': 0.0
        }

    def step(self, actions: np.ndarray) -> VecEnvStepReturn:
        """
        Step the environments with the given action

        :param actions: the action
        :return: observation, reward, done, information
        """
        self.step_async(actions)
        return self.step_wait()

    def step_async(self, actions: np.ndarray) -> None:
        import time
        self._async_start_time = time.time()
        self.actions = actions

        # 阶段1: 所有环境的step1调用总耗时
        self._step1_total = 0.0
        if hasattr(self.envs[0], 'step1') and callable(getattr(self.envs[0], 'step1')):
            step1_start = time.time()
            for env_idx in range(self.num_envs):
                self.envs[env_idx].step1(self.actions[env_idx])
            self._step1_total = time.time() - step1_start
            
            # 阶段2: 50次update循环总耗时
            update_start = time.time()
            for _ in range(50):
                QuadrotorIsaacSim().update()
            self._update_total = time.time() - update_start
        else:
            self._step1_total = 0.0
            self._update_total = 0.0
            
        self._async_total = time.time() - self._async_start_time

    def step_wait(self) -> VecEnvStepReturn:
        import time
        wait_start = time.time()
        
        # 阶段3: 所有env的step调用总耗时
        step_total = 0.0
        for env_idx in range(self.num_envs):
            step_start = time.time()
            obs, self.buf_rews[env_idx], terminated, truncated, self.buf_infos[env_idx] = self.envs[env_idx].step(
                self.actions[env_idx]
            )
            step_total += time.time() - step_start
            
            self.buf_dones[env_idx] = terminated or truncated
            self.buf_infos[env_idx]["TimeLimit.truncated"] = truncated and not terminated
            
            if self.buf_dones[env_idx]:
                self.buf_infos[env_idx]["terminal_observation"] = obs
                obs, self.reset_infos[env_idx] = self.envs[env_idx].reset()
            self._save_obs(env_idx, obs)
        
        self._wait_total = time.time() - wait_start
        
        # 将全局统计注入第一个环境的info
        if self.num_envs > 0:
            self.buf_infos[0]["perf_stats"] = {
                "async_total": self._async_total,
                "step1_total": self._step1_total,
                "update_total": self._update_total,
                "wait_total": self._wait_total,
                "step_total": step_total
            }
        
        # 累加统计数据
        self._time_accumulator['async_total'] += self._async_total
        self._time_accumulator['step1_total'] += self._step1_total
        self._time_accumulator['update_total'] += self._update_total
        self._time_accumulator['wait_total'] += self._wait_total
        self._time_accumulator['step_total'] += step_total
        
        self._step_count += 1
        
        if self._step_count % self._stats_interval == 0:
            self._print_statistics()
            self._reset_stats()
        
        return (self._obs_from_buf(), np.copy(self.buf_rews), np.copy(self.buf_dones), deepcopy(self.buf_infos))

    def _print_statistics(self):
        total = self._stats_interval
        avg = lambda k: self._time_accumulator[k]/total*1000
        
        print(f"\n=== 阶段耗时统计（最近{total}步） ===")
        print(f"异步阶段 | 总:{avg('async_total'):.1f}ms = step1({avg('step1_total'):.1f}ms) + update({avg('update_total'):.1f}ms)")
        print(f"等待阶段 | 总:{avg('wait_total'):.1f}ms = step调用({avg('step_total'):.1f}ms)")
        print(f"单步总耗时: {avg('async_total') + avg('wait_total'):.1f}ms")

    def _reset_stats(self):
        for k in self._time_accumulator:
            self._time_accumulator[k] = 0.0

    def reset(self) -> VecEnvObs:
        for env_idx in range(self.num_envs):
            maybe_options = {"options": self._options[env_idx]} if self._options[env_idx] else {}
            obs, self.reset_infos[env_idx] = self.envs[env_idx].reset(seed=self._seeds[env_idx], **maybe_options)
            self._save_obs(env_idx, obs)
        # Seeds and options are only used once
        self._reset_seeds()
        self._reset_options()
        return self._obs_from_buf()

    def close(self) -> None:
        for env in self.envs:
            env.close()

    def get_images(self) -> Sequence[Optional[np.ndarray]]:
        if self.render_mode != "rgb_array":
            warnings.warn(
                f"The render mode is {self.render_mode}, but this method assumes it is `rgb_array` to obtain images."
            )
            return [None for _ in self.envs]
        return [env.render() for env in self.envs]  # type: ignore[misc]

    def render(self, mode: Optional[str] = None) -> Optional[np.ndarray]:
        """
        Gym environment rendering. If there are multiple environments then
        they are tiled together in one image via ``BaseVecEnv.render()``.

        :param mode: The rendering type.
        """
        return super().render(mode=mode)

    def _save_obs(self, env_idx: int, obs: VecEnvObs) -> None:
        for key in self.keys:
            if key is None:
                self.buf_obs[key][env_idx] = obs
            else:
                self.buf_obs[key][env_idx] = obs[key]  # type: ignore[call-overload]

    def _obs_from_buf(self) -> VecEnvObs:
        return dict_to_obs(self.observation_space, copy_obs_dict(self.buf_obs))

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> List[Any]:
        """Return attribute from vectorized environment (see base class)."""
        target_envs = self._get_target_envs(indices)
        return [getattr(env_i, attr_name) for env_i in target_envs]

    def set_attr(self, attr_name: str, value: Any, indices: VecEnvIndices = None) -> None:
        """Set attribute inside vectorized environments (see base class)."""
        target_envs = self._get_target_envs(indices)
        for env_i in target_envs:
            setattr(env_i, attr_name, value)

    def env_method(self, method_name: str, *method_args, indices: VecEnvIndices = None, **method_kwargs) -> List[Any]:
        """Call instance methods of vectorized environments."""
        target_envs = self._get_target_envs(indices)
        return [getattr(env_i, method_name)(*method_args, **method_kwargs) for env_i in target_envs]

    def env_is_wrapped(self, wrapper_class: Type[gym.Wrapper], indices: VecEnvIndices = None) -> List[bool]:
        """Check if worker environments are wrapped with a given wrapper"""
        target_envs = self._get_target_envs(indices)
        # Import here to avoid a circular import
        from stable_baselines3.common import env_util

        return [env_util.is_wrapped(env_i, wrapper_class) for env_i in target_envs]

    def _get_target_envs(self, indices: VecEnvIndices) -> List[gym.Env]:
        indices = self._get_indices(indices)
        return [self.envs[i] for i in indices]
