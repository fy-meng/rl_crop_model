# Crop Environment for Reinforcement Learning

## Usage
To train the model 
```sh
python train.py
```

By default, this will store the model weights into `./model/dqn.pth`.

This will also create `history_train.npz` in the `output` directory, in which contains:
- `trial`: a vector of shape `(T,)` containing the trial number for each data point, where `T` is the total number of steps;
- `state`: a matrix of shape `(T, N)` containing states at each step, where `N` is the dimension of the state space;
- `action`: a vector of shape `(T,)` containing actions at each step;
- `reward`: a vector of shape `(T,)` containing reward at each step;
- `next_state`: a matrix of shape `(T, N)` containing the next state at each step;
- `q_values_target`: a matrix of shape `(T, M)` containing the Q-values of the target network at each step, where `M` is the dimension of the action space;
- `q_values_target`: a matrix of shape `(T, M)` containing the Q-values of the policy network at each step.

To run the model
```sh
python test.py -t NUM_TRIALS -m MODEL_PATH
```

`MODEL_PATH` is default to `./model/dqn.pth`.

This will also create `history_test.npz` in the `output` directory, in which contains:
- `trial`: a vector of shape `(T,)` containing the trial number for each data point;
- `state`: a matrix of shape `(T, N)` containing states at each step;
- `action`: a vector of shape `(T,)` containing actions at each step;
- `reward`: a vector of shape `(T,)` containing reward at each step;
- `next_state`: a matrix of shape `(T, N)` containing the next state at each step;
- `q_values`: a matrix of shape `(T, M)` containing the Q-values of the target network at each step.

Note the slight difference in keys between the data from training and testing.
