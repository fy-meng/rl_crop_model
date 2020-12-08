from collections import namedtuple
from itertools import count
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim

from crop_model import get_toy_env
from dqn import DQN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


BATCH_SIZE = 64
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 50
TARGET_UPDATE = 10

feature_labels, action_labels, env = get_toy_env()

n_features = len(feature_labels)
# nop, only irrigation, only fertilizer, both
actions = [[False, False], [True, False], [False, True], [True, True]]
n_actions = len(actions)

hidden_layers = [16, 32, 64]

policy_net = DQN(n_features, n_actions, hidden_layers).to(device)
target_net = DQN(n_features, n_actions, hidden_layers).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
buffer = ReplayBuffer(1000)

steps_done = 0


def select_action(q_values):
    global steps_done
    sample = random.random()
    eps = EPS_END + (EPS_START - EPS_END) * np.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps:
        with torch.no_grad():
            # double DQN
            action_id = q_values.argmax()
    else:
        action_id = random.randrange(n_actions)
    return torch.tensor([[action_id]], device=device, dtype=torch.long), \
           torch.tensor(actions[action_id], device=device, dtype=torch.long)


def optimize_model():
    if len(buffer) < BATCH_SIZE:
        return
    transitions = buffer.sample(BATCH_SIZE)
    # Transpose the batch. This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                       if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


episode_durations = []

history = {
    'trial': [],
    'state': [],
    'action': [],
    'reward': [],
    'next_state': [],
    'q_values_target': [],
    'q_values_policy': [],
}

num_episodes = 500
for i_episode in range(num_episodes):
    # Initialize the environment and state
    env.reset()
    state = torch.tensor(env.state_history[0], device=device, dtype=torch.float)
    state = state.unsqueeze(0)
    for t in count():
        q_values_target = target_net(state)
        q_values_policy = policy_net(state)

        # select and perform an action
        action_id, action = select_action(q_values_target)
        next_state, reward, done = env.step(action)
        next_state = torch.tensor(next_state, device=device, dtype=torch.float)
        next_state = next_state.unsqueeze(0)
        reward = torch.tensor([reward], device=device, dtype=torch.float)

        # store the transition in memory
        buffer.push(state, action_id, next_state, reward)

        # store the transition in history
        history['trial'].append(i_episode)
        history['state'].append(state.detach().cpu().numpy().squeeze())
        history['action'].append(action.detach().cpu().numpy())
        history['reward'].append(reward)
        history['next_state'].append(next_state.detach().cpu().numpy().squeeze())
        history['q_values_target'].append(q_values_target.detach().cpu().numpy().squeeze())
        history['q_values_policy'].append(q_values_policy.detach().cpu().numpy().squeeze())

        # move to the next state
        state = next_state

        # perform one step of the optimization (on the target network)
        optimize_model()
        if done:
            episode_durations.append(t + 1)
            break

    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

for k in history.keys():
    history[k] = np.array(history[k])
if not os.path.exists('./output'):
    os.makedirs('./output')
pd.to_pickle(history, './output/history_train.pkl')

if not os.path.exists('./model'):
    os.makedirs('./model')
torch.save(policy_net.state_dict(), './model/dqn.pth')
