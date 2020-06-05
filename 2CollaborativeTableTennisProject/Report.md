# Report

The environment of this project is described in [Readme.md](https://github.com/ht0rohit/Deep-Reinforcement-Learning/blob/master/2CollaborativeTableTennisProject/README.md) file.

### Learning Algorithm 

The environments which require training two or more seperate agents can't be solved by just doing a simple extension of single agent RL to update their policies independently as learning progresses as this causes the environment to appear non-stationary from the viewpoint of any one agent.

Thus, in this project, the agent is trained with [Multi-agent Deep Deterministic Policy Gradients (MADDPG)](https://towardsdatascience.com/openais-multi-agent-deep-deterministic-policy-gradients-maddpg-9d2dad34c82) to reach the desired solution. The MADDPG agent uses multiple DDPG agents which solves the environment by playing against each other. 

* DDPG uses experience replay to sample batches of uncorrelated experiences to learn upon. It distinguishes b/w local and target models for both the actor and critic, similar to fixed Q-targets in DQN technique. Local models are updated by minimizing losses while these values are partially transferred to target models. It's is an off-policy algorithm and can only be used for environments with continuous action spaces.

* DDPG can also be used with CNN to solve environments with higher complexities. It's able to learn good policies with pixel inputs.

                            MADDPG Architecture for two-agent environments

![MADDPG Architecture](https://miro.medium.com/max/1400/1*4aPqpDYFe3ibl0JIO8AeFg.png)

### Model Architecture (HyperParameters)

> Actor / Critic Network

The deep-NN architecture defined in `model.py` consists of 1 network each for Actor (maps state to action) & Critic (maps action to Q-value) based models with 2 hidden layers each consisting of [512, 256] & [256, 128] nodes & with ReLU activation. The output layer in Actor, however, has Tanh activation to clip the output/action to [-1, 1]. The model operates on fully-connected layers.

> Agent

The agent is defined in `agent.py`.

  - `BUFFER_SIZE` = 1e5 # replay buffer size
  - `BATCH_SIZE` = 128  # minibatch size
  - `GAMMA` = 0.99      # reward discount factor
  - `TAU` = 1e-3        # for soft-update of target parameters
  - `LR_ACTOR` = 1e-4   # learning rate for actor
  - `LR_CRITIC` = 1e-3  # learning_rate for critic
  - `WEIGHT_DECAY` = 0	# L2 weight decay
  - `seed` = 0          # for all random choices

> Ornstein-Uhlenbeck Noise

  - `theta` = 0.15
  - `sigma` = 0.2
  
> Training

  - `n_episodes` = 1000                 # max number of episodes
  - `max_t` = while `done = False`      # max step per episode

### Plot of Rewards

The plot contains information about rewards collected & averaged over last 100 episodes. The environment is considered solved when the average reward is atleast +0.5 over the last 100 episodes (epochs). The agent solved the environment in 1766 episodes with an average reward of 0.5053 but can be trained for more episodes to assess if it's actually learning. 

![Plot of Rewards](https://github.com/ht0rohit/Deep-Reinforcement-Learning/blob/master/2CollaborativeTableTennisProject/Assets/rewards.jpg)

### Ideas for Future Work

1. The current implementation has `add_noise = True` set. However, its hyperparameters haven't been tweaked upon much. We could optimize these hyperparameters to get a stable result early on training.

2. We could train the agent for more no. of episodes even after solving the environment to assess when & how sometimes it starts performing horribly with no evidence of recovery, after much no. of episodes.

3. We can solve an environment with a small team of agents instead of just 2 agents. This would give a better understanding for solving complex environments such as [Soccer](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#soccer-twos) or more self created environments from scratch in the future.

4. Various optimizations to the original algorithm such as [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952) & [Hindsight Experience Replay](https://arxiv.org/abs/1707.01495) can be implemented.

5. We can solve an environment with multiple agents (more than two) to get a better understanding of some of the algorithms such as [PPO](https://arxiv.org/pdf/1707.06347.pdf), [A3C](https://arxiv.org/pdf/1602.01783.pdf), and [D4PG](https://openreview.net/pdf?id=SyZipzbCb) that use multiple (non-interacting, parallel) copies of the same agent to distribute the task of gathering experience.
