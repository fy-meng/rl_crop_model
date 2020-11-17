from typing import List, Callable

import numpy as np


class CropEnv:
    def __init__(self, num_features: int, num_actions: int, continuous: bool,
                 transition_funcs: List[Callable], initial_state=None,
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
        transition_funcs : list of callable
            List of transition functions. Length should equal to
            `num_features`. Each function should take in two parameters,
            `state_history` and `action_history`, where `state_history` is
            a list of `t + 1` `np.ndarray` of shape `(num_features,)`, and
            `action_history` is a list of `t` numbers, in which `t` is the
            total number of steps run.
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
        assert len(transition_funcs) == num_features
        self.transition_funcs = transition_funcs

        if initial_state is not None:
            assert np.array(initial_state).shape == (num_features,)
            self.initial_state = np.array(initial_state)

        if feature_labels is not None:
            assert len(feature_labels) == num_features
        self.feature_labels = feature_labels

        if action_labels is not None:
            assert len(action_labels) == num_actions
        self.action_labels = action_labels

        self.state_history = [self.initial_state]
        self.action_history = []

    def restart(self):
        """Restart the environment.
        Initialize to `self.initial_state`, or all zero if
        `self.initial_state` is `None`.
        """
        self.state_history = [self.initial_state]
        self.action_history = []

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
        """
        self.action_history.append(action)
        new_state = self.state_history[-1].copy()
        for i in range(self.num_features):
            new_state[i] = self.transition_funcs[i](self.state_history, self.action_history)
        self.state_history.append(new_state)
        return new_state

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
            print()


def get_toy_env() -> CropEnv:
    # the model is:
    # weather -> precipitation -> water -> yield
    # watering (action) -> water -> yield
    # soil -> yield
    # fertilizer -> yield
    # crop_species -> yield
    feature_labels = ['weather', 'precipitation', 'water', 'soil', 'yield']
    action_labels = ['watering', 'fertilizer']

    def weather_transition(state_history, _action_history, p):
        # weather is good (0) or bad (1)
        # weather has a chance of `1 - p` to change
        prev_weather = state_history[-1][0]
        return prev_weather if np.random.rand() < p else 1 - prev_weather

    def precipitation_transition(state_history, _action_history,
                                 mean_lo, std_lo, mean_hi, std_hi):
        # precipitation is a float
        # if previous weather is good, return max(N(`mean_lo`, `std_lo`), 0)
        # otherwise, return max(N(`mean_hi`, `std_hi`), 0)
        prev_weather = state_history[-1][0]
        if prev_weather == 0:
            precipitation = np.random.normal(mean_lo, std_lo)
        else:
            precipitation = np.random.normal(mean_hi, std_hi)
        precipitation = max(precipitation, 0)
        return precipitation

    def water_transition(state_history, action_history):
        # amount of water crop recieved is precipitation + manual watering
        manual_watering = action_history[-1][0] if len(action_history) else 0
        return state_history[-1][1] + manual_watering

    def soil_transition(state_history, action_history,
                        regular_decay, fertilizer_decay):
        # soil degrade with regular_decay * fertilizer_decay
        soil = state_history[-1][3] * regular_decay
        if action_history[-1][1]:
            soil *= fertilizer_decay
        return soil

    def yield_transition(state_history, action_history, water_mean, water_std, fertilizer_bonus):
        # yield = water_contribution + soil + fertilizer_bonus
        # water_contribution = N(water_mean, water_std)
        # note that water_contribution could be negative
        # fertilizer_bonus only applies if fertilized
        return np.random.normal(water_mean, water_std) \
               + state_history[-1][3] \
               + fertilizer_bonus if action_history[-1][1] else 0

    toy_transition_funcs = [
        # weather
        lambda states, actions: weather_transition(states, actions, 0.6),
        # precipitation
        lambda states, actions: precipitation_transition(states, actions, 20, 10, 200, 50),
        # water
        water_transition,
        # soil
        lambda states, actions: soil_transition(states, actions, 0.95, 0.9),
        # yield
        lambda states, actions: yield_transition(states, actions, 100, 20, 50)
    ]
    env = CropEnv(5, 2, False, toy_transition_funcs,
                  initial_state=np.array([0, 0, 0, 20, 0]),
                  feature_labels=feature_labels, action_labels=action_labels)
    return env
