import os
from typing import *
import numpy as np
import gym
from sklearn.preprocessing import StandardScaler

# ENV_NAME = 'Acrobot-v1'
# env = gym.make(ENV_NAME)
#
# print(f"action_space = {env.action_space.n}, observation_space: {env.observation_space}")


def get_maximum_environments_space_and_action_size(available_environments_names: List[object] = None) -> Tuple[int, int]:
    """
    returns the maximum state space and action space size for all available environments.
    :param available_environments_names: list of names of environments as configured for gym package
    :return: (int, int) - max state space size, max action space size
    """
    available_environments_names = available_environments_names if available_environments_names is not None else \
        ['CartPole-v1', 'Acrobot-v1', 'MountainCarContinuous-v0']
    max_state_size = 0
    max_action_size = 0
    for env_name in available_environments_names:
        env = gym.make(env_name)
        env_state_size = len(env.reset())
        try:
            env_action_size = env.action_space.n
        except AttributeError:
            env_action_size = env.action_space.bounded_above.shape[0]
        max_state_size = env_state_size if env_state_size > max_state_size else max_state_size
        max_action_size = env_action_size if env_action_size > max_action_size else max_action_size

    return max_state_size, max_action_size


def fill_space_vector_with_zeros(org_space_vector: Union[np.array], max_size_of_space_vector: int) -> np.array:
    """
    appends to the end of the vector 'org_space_vector' 0's up to size of 'max_size_of_space_vector'.
    :param org_space_vector: 1D array
    :param max_size_of_space_vector: int
    :return:
    """
    filled_array = np.zeros(max_size_of_space_vector)
    filled_array[:len(org_space_vector)] = org_space_vector
    return filled_array


def sample_from_env(env, idx_to_sample: int = 0, n_samples: int = 10000):
    samples = np.array(
        [env.observation_space.sample()[0][idx_to_sample] for x in range(n_samples)])

    return samples


def get_scaler(space_min: float = 0, space_max: float = 0,
               method: str = 'min_max',
               env_to_sample=None,
               idx_to_sample: int = 0,
               n_samples: int = 10000):
    mu = (space_max - space_min) / 2 if env_to_sample is None else np.mean(sample_from_env(env_to_sample))
    sigma = np.sqrt(((space_max-mu)**2 + (space_min-mu)**2)/2) if env_to_sample is None else np.std(sample_from_env(env_to_sample))

    scaler = None
    if env_to_sample is not None:
        scaler = StandardScaler()
        scaler.fit(sample_from_env(env_to_sample, idx_to_sample=idx_to_sample, n_samples=n_samples))

    def scale_by_min_max(val: float):
        scaled_val = (space_max - val)/(space_max-space_min)
        return scaled_val

    def scale_by_standard_with_samples(val: float):
        if scaler is None:
            raise ValueError(f"Must include an environment to sample!")

        return scaler.fit_transform([val]).reshape(-1)

    def scale_by_standard_with_min_max(val: float):
        return (val-mu) / sigma

    if method == 'min_max':
        return scale_by_min_max
    elif method == 'standard':
        return scale_by_standard_with_min_max if env_to_sample is None else scale_by_standard_with_samples
    else:
        raise f"method of scaling must be either 'min_max' or 'standard', but got {method}"


def get_env_min_max_action_space(env):
    env_action_space_min = env.action_space.low[0]
    env_action_space_max = env.action_space.high[0]
    return env_action_space_min, env_action_space_max