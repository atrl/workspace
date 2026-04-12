from __future__ import annotations

import sys
from typing import Tuple, cast

import gymnasium as _gym
import numpy as np
import tianshou.env.venvs as _tv
from qlib.rl.utils import EnvWrapper as _EW
from qlib.rl.utils.finite_env import FiniteVectorEnv as _FVE


def patch_gym_compat() -> None:
    if getattr(patch_gym_compat, "_done", False):
        return
    if "gym" not in sys.modules:
        sys.modules["gym"] = _gym
        sys.modules["gym.spaces"] = _gym.spaces

    orig_ew_reset = _EW.reset
    orig_ew_step = _EW.step
    orig_fve_reset = _FVE.reset

    def _ew_reset(self, **kwargs):
        obs = orig_ew_reset(self, **kwargs)
        return obs, {}

    def _ew_step(self, action, **kwargs):
        obs, rew, done, info = orig_ew_step(self, action, **kwargs)
        return obs, rew, done, False, info

    def _fve_reset(self, id=None, **kwargs):
        obs = orig_fve_reset(self, id=id, **kwargs)
        return obs, {}

    def _fve_step(self, action, id=None):
        self._assert_is_not_closed()
        id = self._wrap_id(id)
        if isinstance(action, np.ndarray) and action.shape:
            assert len(action) == len(id)
        else:
            action = [action] * len(id)
        for env_id, act in zip(id, action):
            self.workers[env_id].send(act)
        result = []
        for i in id:
            r = self.workers[i].recv()
            if len(r) == 5:
                obs, rew, done, truncated, info = r
                done = done or truncated
            else:
                obs, rew, done, info = r
            result.append((obs, rew, done, info))
            for logger in self._logger:
                logger.on_env_step(i, obs, rew, done, info)
        for i, r in zip(id, result):
            if i in self._alive_env_ids and r[2]:
                self._alive_env_ids.remove(i)
        obs_arr = np.stack([r[0] for r in result])
        rew_arr = np.stack([r[1] for r in result])
        done_arr = np.stack([r[2] for r in result])
        info_arr = np.stack([r[3] for r in result])
        return cast(
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
            (obs_arr, rew_arr, done_arr, np.zeros_like(done_arr), info_arr),
        )

    _EW.reset = _ew_reset
    _EW.step = _ew_step
    _FVE.reset = _fve_reset
    _FVE.step = _fve_step
    _tv._patch_env_generator = lambda fn: fn
    patch_gym_compat._done = True
