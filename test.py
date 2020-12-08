import argparse
import os

import numpy as np
import pandas as pd
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
    history = {
        'trial': [],
        'state': [],
        'action': [],
        'reward': [],
        'next_state': [],
        'done': [],
        'q_values': [],
    }

    # DQN agent
    for i_episode in range(num_trials):
        env.reset()
        state = torch.tensor(env.state_history[0], device=device, dtype=torch.float)
        done = False
        while not done:
            q_values = net(state)
            action_id = q_values.argmax()
            action = actions[action_id]
            next_state, reward, done = env.step(action)

            # store history
            history['trial'].append(i_episode)
            history['state'].append(state.detach().cpu().numpy().squeeze())
            history['action'].append(action)
            history['reward'].append(reward)
            history['next_state'].append(next_state)
            history['done'].append(done)
            history['q_values'].append(q_values.detach().cpu().numpy())

            state = torch.tensor(next_state, device=device, dtype=torch.float)
        print(f'trail {i_episode:05d}:\n\tsteps = {len(env.state_history)}\n\treturn = {env.total_return()}')

    for k in history.keys():
        history[k] = np.array(history[k])
    if not os.path.exists('./output'):
        os.makedirs('./output')
    pd.to_pickle(history, './output/history_test.pkl')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', action='store', dest='num_trials', type=int, default=1,
                        help='number of trials')
    parser.add_argument('-m', action='store', dest='model_path', type=str, default='./model/dqn.pth',
                        help='model file path')
    args = parser.parse_args()

    test(args.num_trials, args.model_path)
