import numpy as np
import matplotlib.pyplot as plt
import torch

from crop_model import CropEnv, get_toy_env
from dqn import DQN
from sarfa_saliency import computeSaliencyUsingSarfa

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

feature_labels, action_labels, env = get_toy_env()
env: CropEnv = env

n_features = len(feature_labels)
# nop, only irrigation, only fertilizer, both
actions = [[False, False], [True, False], [False, True], [True, True]]
n_actions = len(actions)

hidden_layers = [32, 64]

net = DQN(n_features, n_actions, hidden_layers).to(device)
net.load_state_dict(torch.load('./model/dqn.pth'))

# nop, only irrigation, only fertilizer, both
actions = [[False, False], [True, False], [False, True], [True, True]]

state = torch.tensor(env.state_history[0], device=device, dtype=torch.float)
done = False
while not done:
    action_id = net(state).argmax()
    action = actions[action_id]
    state, _, done = env.step(action)
    state = torch.tensor(state, device=device, dtype=torch.float)

# plot state history
state_history = np.array(env.state_history)
for feature_history, label in zip(state_history.T, feature_labels):
    if label == 'weather':
        feature_history = feature_history * 100
    plt.plot(feature_history, label=label)
plt.legend()
plt.xticks([0, 5, 10, 15, 20])
plt.savefig('./imgs/state_history.png')
plt.clf()

# plot saliency
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
        q_values /= max_q

        for j in range(n_features):
            state_perturbed = state.clone()

            if j == 0 or j == 3:  # boolean values, perturb by flipping
                state_perturbed[j] = not state_perturbed[j]
            else:  # numerical values, perturb to median
                state_perturbed[j] = medians[j]

            q_values_perturbed = net(state_perturbed)
            q_values_perturbed = np.array(q_values_perturbed)
            q_values_perturbed /= max_q

            q_dict = {i: q for i, q in enumerate(q_values)}
            q_perturbed_dict = {i: q for i, q in enumerate(q_values_perturbed)}

            saliency = computeSaliencyUsingSarfa(action_id, q_dict, q_perturbed_dict)[0]
            saliency_history[i, j] += saliency

plt.stackplot(np.arange(len(saliency_history)), saliency_history.T, labels=feature_labels)
plt.legend()
plt.xticks([0, 5, 10, 15, 20])
plt.savefig('./imgs/saliency_history.png')
plt.clf()
