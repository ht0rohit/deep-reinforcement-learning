# Report

The environment of this project is described in [Readme.md](https://github.com/ht0rohit/Deep-Reinforcement-Learning/blob/master/0ContinuousControl/README.md) file.

### Learning Algorithm

In this project, the agent is trained with [Deep Deterministic Policy Gradients (DDPG)](https://towardsdatascience.com/deep-deterministic-policy-gradients-explained-2d94655a9b7b). 

The Actor-Critic learning algorithm is used to represent the policy function independently of the value function. The policy function structure is known as the actor, and the value function structure is referred to as the critic. The actor produces an action given the current state of the environment, and the critic produces a TD (Temporal-Difference) error signal given the state and resultant reward. The output of the critic drives learning in both the actor and the critic.

DDPG uses experience replay to sample batches of uncorrelated experiences to learn upon. It distinguishes b/w local and target models for both the actor and critic, similar to fixed Q-targets in DQN technique. Local models are updated by minimizing losses while these values are partially transferred to target models. It's is an off-policy algorithm and can only be used for environments with continuous action spaces.

DDPG can also be used with CNN to solve environments with higher complexities. It's able to learn good policies with pixel inputs.

![DDPGAlgo](https://miro.medium.com/max/1084/1*BVST6rlxL2csw3vxpeBS8Q.png)

### Model Architecture (HyperParameters)

> Actor / Critic Network

The deep-NN architecture defined in `model.py` consists of 1 network each for Actor (maps state to action) & Critic (maps action to Q-value)  based models with 2 hidden layers consisting of [256, 256] nodes & with ReLU activation & Batch Normalization. The output layer in Actor, however, has Tanh activation to clip the output/action to [-1, 1]. The model operates on fully-connected layers.

> Agent

The agent is defined in `agent.py`.

  - `BUFFER_SIZE` = 1e5 # replay buffer size
  - `BATCH_SIZE` = 64   # minibatch size
  - `GAMMA` = 0.99      # reward discount factor
  - `TAU` = 1e-3        # for soft-update of target parameters
  - `LR_ACTOR / LR_CRITIC` = 5e-4         # the learning_rate from previous project experience proved quite useful
  - `UPDATE_EVERY` = 4  # how often to update the network
  - `WEIGHT_DECAY` = 0	# L2 weight decay
  - `seed` = 0          # for all random choices

> Ornstein-Uhlenbeck Noise

  - `theta` = 0.15
  - `sigma` = 0.2
  
> Training

  - `n_episodes` = 1000 # max number of episodes
  - `max_t` = 1000      # max step per episode

### Plot of Rewards

The plot contains information about rewards collected & averaged over last 100 episodes. The environment is considered solved when the average reward is atleast +30 over the last 100 episodes (epochs). The agent solved the environment in 403 episodes with an average reward of 30.07 but can be trained for more episodes to assess if it's actually learning. 

![Plot of Rewards](https://github.com/ht0rohit/Deep-Reinforcement-Learning/blob/master/1ContinuousControl/Assets/rewards.PNG)

### Ideas for Future Work

1. We can Optimize various hyper-parameters to solve the environent in fewer episodes.

2. The current implementation has `add_noise = False` set. We can instead add Ornsetein-Uhlenbeck noise to the action space.

3. We can solve an environment with multiple agents to get a better understanding of some of the algorithms such as [PPO](https://arxiv.org/pdf/1707.06347.pdf), [A3C](https://arxiv.org/pdf/1602.01783.pdf), and [D4PG](https://openreview.net/pdf?id=SyZipzbCb) that use multiple (non-interacting, parallel) copies of the same agent to distribute the task of gathering experience.
