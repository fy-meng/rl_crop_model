from typing import List, Callable

import numpy as np


class CropEnv:
    def __init__(self, num_features: int, num_actions: int, continuous: bool,
                 transition_func: Callable, stop_cond: Callable, initial_state=None,
                 feature_labels: List[str] = None, action_labels: List[str] = None):
        """Initialize a crop experiment environment.
        Parameters
        ----------
        num_features : int
            Number of features.
        num_actions : int
            Dimension of the action space.
        continuous : bool
            If the action space is continuous or not.
        transition_func : callable
            Transition functions that take in three parameters: `state_history`
            `action_history` and `reward_history`, and returns the next state
            and reward. `state_history` is list of `t + 1` `np.ndarray` of shape
            `(num_features,)`, `action_history` and `reward_history` are lists
            of `t` numbers, in which `t` is the total number of steps run.
        stop_cond : callable
            A function that takes in the current state and the iteration
            number and return `True` if the simulation should terminate.
        initial_state : , (n,) array, or `None`, optional
            Initial state. Also used to restart the environment. `n` is
            equal to `num_features`.
        feature_labels : list of string, optional
            List of feature labels. Length should equal to `num_features`.
        action_labels : list of string, optional
            List of action labels. Length should equal to `num_actions`.
        """
        self.num_features = num_features
        self.num_actions = num_actions
        self.continuous = continuous
        self.transition_func = transition_func
        self.stop_cond = stop_cond
        self.random_seed = None

        if initial_state is not None:
            assert np.array(initial_state).shape == (num_features,)
            self.initial_state = np.array(initial_state)

        if feature_labels is not None:
            assert len(feature_labels) == num_features
        self.feature_labels = feature_labels

        if action_labels is not None:
            assert len(action_labels) == num_actions
        self.action_labels = action_labels

        self.iter = 0
        self.state_history = [self.initial_state]
        self.action_history = []
        self.reward_history = []
        self.reset()

    def set_random_seed(self, random_seed):
        """Set the random seed. Also reset the environment.
        Parameters
        ----------
        random_seed : int
            The random seed.
        """
        self.random_seed = random_seed
        self.reset()

    def reset(self):
        """Restart the environment.
        Initialize to `self.initial_state`, or all zero if
        `self.initial_state` is `None`. If `self.random_seed` is not None, also
        set the random seed to that value.
        """
        self.iter = 0
        self.state_history = [self.initial_state]
        self.action_history = []
        self.reward_history = []
        if self.random_seed is not None:
            np.random.seed(self.random_seed)

    def step(self, action: np.ndarray) -> np.ndarray:
        """Execute one step in the environment.
        Parameters
        ----------
        action : (m,) array
            Action to execute where `m` is equal to `num_actions`.
        Returns
        ----------
        next_state: (n,) array
            Next state after executing the action. `n` is equal to
            `num_features`.
        reward: float
            Reward for the action.
        done: bool
            `True` if the simulation is ended.
        """
        self.iter += 1

        self.action_history.append(action)
        new_state, reward = self.transition_func(self.state_history,
                                                 self.action_history,
                                                 self.reward_history)
        self.state_history.append(new_state)
        self.reward_history.append(reward)
        return new_state, reward, self.stop_cond(new_state, self.iter)

    def total_return(self):
        """Compute the return of the current run.
        Returns
        -------
        reward_sum : float
            Sum of the rewards of the current run.
        """
        return sum(self.reward_history)

    def print_history(self):
        for t in range(len(self.state_history)):
            print(f'at step = {t}:')
            if self.feature_labels is not None:
                state_str = ", ".join([f"{label}={val}" for val, label
                                       in zip(self.state_history[t], self.feature_labels)])
            else:
                state_str = str(self.state_history[t])
            print(f'state:\n\t{state_str}')

            if t < len(self.action_history):
                if self.action_labels is not None:
                    action_str = ", ".join([f"{label}={val}" for val, label
                                            in zip(self.action_history[t], self.action_labels)])
                else:
                    action_str = str(self.action_history[t])
                print(f'action:\n\t{action_str}')
                print(f'reward: {self.reward_history[t]}')
            print()


def get_toy_env():
    # the model is:
    # weather -> precipitation -> water -> yield
    # watering (action) -> water -> yield
    # soil -> yield
    # fertilizer -> yield
    # crop_species -> yield
    feature_labels = ['weather', 'precipitation', 'water', 'soil']
    action_labels = ['irrigation', 'fertilizer']

    def transition_func(state_history, action_history, _reward_history):
        WEATHER_CHANGE_PROB = 0.2
        GOOD_WEATHER_PRECIP_MEAN = 0
        GOOD_WEATHER_PRECIP_STD = 50
        BAD_WEATHER_PRECIP_MEAN = 150
        BAD_WEATHER_PRECIP_STD = 50
        IRRIGATION_MEAN = 100
        IRRIGATION_STD = 10
        SOIL_DECAY = 0.95
        FERTILIZER_DECAY = 0.8
        BASE_YIELD_MEAN = 100
        BASE_YIELD_STD = 20
        DESIRED_WATER = 100
        WATER_PENALTY = 0.02
        FERTILIZER_BONUS = 3.5

        prev_state = state_history[-1]
        action = action_history[-1]

        prev_weather, prev_soil = prev_state[0], prev_state[3]
        irrigate, fertilize = action

        # weather is good (0) or bad (1)
        # weather has a chance of `1 - p` to change
        weather = 1 - prev_weather if np.random.rand() < WEATHER_CHANGE_PROB else prev_weather

        # if previous weather is good, precipitation is max(N(`mean_lo`, `std_lo`), 0)
        # otherwise, return max(N(`mean_hi`, `std_hi`), 0)
        if prev_weather == 0:
            precip = np.random.normal(GOOD_WEATHER_PRECIP_MEAN, GOOD_WEATHER_PRECIP_STD)
        else:
            precip = np.random.normal(BAD_WEATHER_PRECIP_MEAN, BAD_WEATHER_PRECIP_STD)
        precip = max(precip, 0)

        # amount of water crop received is precipitation + irrigation
        irrigation = np.random.normal(IRRIGATION_MEAN, IRRIGATION_STD) if irrigate else 0
        water = precip + irrigation

        # soil degrade with regular_decay * fertilizer_decay
        soil = prev_soil * SOIL_DECAY
        if fertilize:
            soil *= FERTILIZER_DECAY

        # yield = max((base_yield + water_contribution) * soil * fertilizer_bonus, 0)
        # water_penalty = -water_penalty * (water - water_base) ** 2
        # note that water_contribution could be negative
        # fertilizer_bonus only applies if fertilized
        reward = np.random.normal(BASE_YIELD_MEAN, BASE_YIELD_STD)
        reward -= WATER_PENALTY * ((water - DESIRED_WATER) ** 2)
        reward *= soil
        if fertilize:
            reward *= FERTILIZER_BONUS
        reward = max(reward, 0)

        new_state = np.array([weather, precip, water, soil])
        return new_state, reward

    def stop_cond(state, _it):
        # terminate if soil is < 0.1
        return state[3] < 0.1

    env = CropEnv(4, 2, False, transition_func, stop_cond,
                  initial_state=np.array([0, 0, 0, 1.]),
                  feature_labels=feature_labels, action_labels=action_labels)
    return feature_labels, action_labels, env
