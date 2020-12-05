import numpy as np
import matplotlib.pyplot as plt
import torch

from crop_model import CropEnv, get_toy_env
from dqn import DQN
from sarfa_saliency import computeSaliencyUsingSarfa

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

feature_labels, action_labels, env = get_toy_env()
env: CropEnv = env
env.set_random_seed(42)

n_features = len(feature_labels)
# nop, only irrigation, only fertilizer, both
actions = [[False, False], [True, False], [False, True], [True, True]]
n_actions = len(actions)

hidden_layers = [16, 32, 64]

net = DQN(n_features, n_actions, hidden_layers).to(device)
net.load_state_dict(torch.load('./model/dqn.pth'))


def plot_state_history(env, path):
    state_history = np.array(env.state_history)
    for feature_history, label in zip(state_history.T, feature_labels):
        if label == 'weather' or label == 'soil':
            feature_history = feature_history * 100
        plt.plot(feature_history, label=label)
    plt.xlabel('iteration')
    plt.legend()
    plt.savefig(path)
    plt.clf()


# nop, only irrigate, only fertilize, both
actions = [[False, False], [True, False], [False, True], [True, True]]

# simulate do nothing agent
done = False
while not done:
    action = [False, False]
    _, _, done = env.step(action)
print(f'do nothing agent:\n\tsteps = {len(env.state_history)}\n\treturn = {env.total_return()}')
plot_state_history(env, './imgs/state_history_do_nothing.png')
env.reset()

# simulate human agent
state = env.state_history[0]
done = False
while not done:
    if state[0] == 0:  # good weather, irrigate and fertilize
        action = [True, True]
    else:  # bad weather, do nothing
        action = [False, False]
    state, _, done = env.step(action)
print(f'human agent:\n\tsteps = {len(env.state_history)}\n\treturn = {env.total_return()}')
plot_state_history(env, './imgs/state_history_human.png')
env.reset()

# store history
state_history = []
action_history = []
reward_history = []
next_state_history = []
q_values_history = []

# DQN agent
state = torch.tensor(env.state_history[0], device=device, dtype=torch.float)
done = False
while not done:
    q_values = net(state)
    action_id = q_values.argmax()
    action = actions[action_id]
    next_state, reward, done = env.step(action)

    # store history
    state_history.append(state)
    action_history.append(action)
    reward_history.append(reward)
    next_state_history.append(next_state)
    q_values_history.append(q_values)

    state = torch.tensor(next_state, device=device, dtype=torch.float)

history_dict = {
    'state': state_history,
    'action': action_history,
    'reward': reward_history,
    'next_state': next_state_history,
    'q_values': q_values_history
}
np.savez('./output/history_test.npz', **history_dict)

print(f'DQN agent:\n\tsteps = {len(env.state_history)}\n\treturn = {env.total_return()}')

plot_state_history(env, './imgs/state_history_dqn.png')

# plot saliency
state_history = np.array(env.state_history)
saliency_history = np.zeros_like(state_history[:-1])
saliency_history = saliency_history.astype(np.float64)
medians = np.median(state_history, axis=0)

with torch.no_grad():
    for i, state in enumerate(state_history[:-1]):
        state = torch.tensor(state, device=device, dtype=torch.float)
        q_values = net(state)
        q_values = np.array(q_values)
        action_id = q_values.argmax()

        max_q = np.max(q_values)
        if max_q:
            q_values /= max_q

        for j in range(n_features):
            state_perturbed = state.clone()

            if j == 0 or j == 3:  # boolean values, perturb by flipping
                state_perturbed[j] = not state_perturbed[j]
            else:  # numerical values, perturb to median
                state_perturbed[j] = medians[j]

            q_values_perturbed = net(state_perturbed)
            q_values_perturbed = np.array(q_values_perturbed)
            if max_q:
                q_values_perturbed /= max_q

            q_dict = {i: q for i, q in enumerate(q_values)}
            q_perturbed_dict = {i: q for i, q in enumerate(q_values_perturbed)}

            saliency = computeSaliencyUsingSarfa(action_id, q_dict, q_perturbed_dict)[0]
            saliency_history[i, j] += saliency

plt.stackplot(np.arange(len(saliency_history)), saliency_history.T, labels=feature_labels)
plt.legend()
plt.savefig('./imgs/saliency_history.png')
plt.clf()
