import argparse

import numpy as np
import torch

from crop_model import CropEnv, get_toy_env
from dqn import DQN


def test(num_trials, model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    feature_labels, action_labels, env = get_toy_env()
    env: CropEnv = env

    n_features = len(feature_labels)
    # nop, only irrigation, only fertilizer, both
    actions = [[False, False], [True, False], [False, True], [True, True]]
    n_actions = len(actions)

    hidden_layers = [16, 32, 64]

    net = DQN(n_features, n_actions, hidden_layers).to(device)
    net.load_state_dict(torch.load(model_path))

    # actions:
    # nop, only irrigate, only fertilize, both
    actions = [[False, False], [True, False], [False, True], [True, True]]

    # store history
    trial_history = []
    state_history = []
    action_history = []
    reward_history = []
    next_state_history = []
    q_values_history = []

    # DQN agent
    for i in range(num_trials):
        env.reset()
        state = torch.tensor(env.state_history[0], device=device, dtype=torch.float)
        done = False
        while not done:
            q_values = net(state)
            action_id = q_values.argmax()
            action = actions[action_id]
            next_state, reward, done = env.step(action)

            # store history
            trial_history.append(i)
            state_history.append(state.detach().numpy())
            action_history.append(action)
            reward_history.append(reward)
            next_state_history.append(next_state)
            q_values_history.append(q_values.detach().numpy())

            state = torch.tensor(next_state, device=device, dtype=torch.float)
        print(f'trail {i:05d}:\n\tsteps = {len(env.state_history)}\n\treturn = {env.total_return()}')

    history_dict = {
        'trial': trial_history,
        'state': state_history,
        'action': action_history,
        'reward': reward_history,
        'next_state': next_state_history,
        'q_values': q_values_history
    }
    np.savez('./output/history_test.npz', **history_dict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', action='store', dest='num_trials', type=int, default=1,
                        help='number of trials')
    parser.add_argument('-m', action='store', dest='model_path', type=str, default='./model/dqn.pth',
                        help='model file path')
    args = parser.parse_args()

    test(args.num_trials, args.model_path)
