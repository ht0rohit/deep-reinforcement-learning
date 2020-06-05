import numpy as np
import random, copy
from collections import namedtuple, deque

import torch
import torch.nn.functional as F
import torch.optim as optim

from model import Actor, Critic


BUFFER_SIZE = int(1e5)	# replay buffer size
BATCH_SIZE = 128		# minibatch size
GAMMA = 0.99			# discount factor
TAU = 1e-3				# for soft update of target parameters
LR_ACTOR = 1e-4			# learning rate of the actor 
LR_CRITIC = 1e-3		# learning rate of the critic
WEIGHT_DECAY = 0		# L2 weight decay

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
			


class ReplayBuffer(object):
	"""Fixed-size buffer to store experience tuples."""

	def __init__(self, buffer_size, batch_size, num_agents, seed):
		"""Initialize a ReplayBuffer object.

		Params:
			action_size (int) ---  dimension of each action
			buffer_size (int) ---  maximum size of buffer
			batch_size (int)  ---  size of each training batch
			seed (int)		  ---  random seed
		"""
		self.num_agents = num_agents
		self.memory = deque(maxlen = buffer_size)  
		self.batch_size = batch_size
		self.experience = namedtuple("Experience", field_names = ["states", "actions", "rewards", "next_states", "dones"])
		random.seed(seed)
		
		
	def add(self, state, action, reward, next_state, done):
		"""Add a new experience to memory."""
		e = self.experience(state, action, reward, next_state, done)
		self.memory.append(e)
	
	
	def sample(self):
		"""Randomly sample a batch of experiences from memory."""
		
		experiences = random.sample(self.memory, k = self.batch_size)
		
		# Note that states, actions, next_states are now a list containing values from each agent
		states = [torch.from_numpy(np.vstack([e.states[index] for e in experiences if e is not None])).float().to(device) for index in range(self.num_agents)]
		actions = [torch.from_numpy(np.vstack([e.actions[index] for e in experiences if e is not None])).float().to(device) for index in range(self.num_agents)]
		rewards = torch.from_numpy(np.vstack([e.rewards for e in experiences if e is not None])).float().to(device)	
		next_states = [torch.from_numpy(np.vstack([e.next_states[index] for e in experiences if e is not None])).float().to(device) for index in range(self.num_agents)]	 
		dones = torch.from_numpy(np.vstack([e.dones for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

		return (states, actions, rewards, next_states, dones)


	def __len__(self):
		"""Return the current size of internal memory."""
		return len(self.memory)
	
#Replay memory
sharedMem = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, num_agents = 2, seed = 0)



class Agent(object):
	"""Interacts with and learns from the environment."""
	
	def __init__(self, state_size, action_size, seed):
		"""Initialize an Agent object.
		
		Params :
			state_size (int)  --- dimension of each state
			action_size (int) --- dimension of each action
			seed (int)		  --- random seed
		"""
		
		self.state_size = state_size
		self.action_size = action_size
		random.seed(seed)
		
		# Construct Actor networks
		self.actor_local = Actor(state_size, action_size, seed).to(device)
		self.actor_target = Actor(state_size, action_size, seed).to(device)
		self.actor_optimizer = optim.Adam(self.actor_local.parameters(),lr = LR_ACTOR)

		# Construct Critic networks 
		self.critic_local = Critic(state_size, action_size , seed).to(device)
		self.critic_target = Critic(state_size, action_size , seed).to(device)
		self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr = LR_CRITIC, weight_decay = WEIGHT_DECAY)

		# noise processing
		self.noise = OUNoise(action_size, seed)
		
			
			
	def step(self):
		"""Sample experiences from a shared Replay buffer & update policy
		using the learned samples"""
	
		if len(sharedMem) > BATCH_SIZE:
			experiences = sharedMem.sample()
			self.learn(experiences, GAMMA)		  


	def act(self, state, add_noise = True):
		"""Returns actions for given state as per current policy."""
		state = torch.from_numpy(state).float().to(device)
		
		self.actor_local.eval()
		with torch.no_grad():
			action = self.actor_local(state).cpu().data.numpy()
		self.actor_local.train()
		
		if add_noise:
			action += self.noise.sample()
		return np.clip(action, -1, 1)


	def reset(self):
		"""Reset the noise when 'add_noise = True' is set"""
		self.noise.reset()


	def learn(self, experiences, gamma):
		"""Update policy and value parameters using given batch of experience tuples.
		Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
		where,
			actor_target(state) -> action
			critic_target(state, action) -> Q-value

		Params:
			experiences (Tuple[torch.Tensor]) ---  tuple of (s, a, r, s', done) tuples 
			gamma (float)					  ---  discount factor
		"""
		states_list, actions_list, rewards, next_states_list, dones = experiences
					
		next_states_tensor = torch.cat(next_states_list, dim = 1).to(device)
		states_tensor = torch.cat(states_list, dim = 1).to(device)
		actions_tensor = torch.cat(actions_list, dim = 1).to(device)
		
		# ---------------------------- update critic ---------------------------- #
		# Get predicted next-state actions and Q values from target models
		next_actions = [self.actor_target(states) for states in states_list]		
		next_actions_tensor = torch.cat(next_actions, dim=1).to(device)		   
		Q_targets_next = self.critic_target(next_states_tensor, next_actions_tensor)		
		# Compute Q targets for current states (y_i)
		Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))		
		# Compute critic loss
		Q_expected = self.critic_local(states_tensor, actions_tensor)
		critic_loss = F.mse_loss(Q_expected, Q_targets)		   
		# Minimize the loss
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		#torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
		self.critic_optimizer.step()

		# ---------------------------- update actor ---------------------------- #
		# Compute actor loss
		# take the current states and predict actions
		actions_pred = [self.actor_local(states) for states in states_list]		   
		actions_pred_tensor = torch.cat(actions_pred, dim=1).to(device)
		# -1 * (maximize) Q value for the current prediction
		actor_loss = -self.critic_local(states_tensor, actions_pred_tensor).mean()		  
		# Minimize the loss
		self.actor_optimizer.zero_grad()
		actor_loss.backward()		 
		#torch.nn.utils.clip_grad_norm_(self.actor_local.parameters(), 1)
		self.actor_optimizer.step()

		# ----------------------- update target networks ----------------------- #
		self.soft_update(self.critic_local, self.critic_target, TAU)
		self.soft_update(self.actor_local, self.actor_target, TAU)					   


	def soft_update(self, local_model, target_model, tau):
		"""Soft update model parameters.
		θ_target = τ*θ_local + (1 - τ)*θ_target

		Params:
			local_model	 --- PyTorch model (weights will be copied from)
			target_model --- PyTorch model (weights will be copied to)
			tau (float)	 --- interpolation parameter 
		"""
		for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
			target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
			
			
			
class OUNoise(object):
	"""Ornstein-Uhlenbeck process."""

	def __init__(self, size, seed, mu = 0., theta = 0.15, sigma = 0.2):
		"""Initialize parameters and noise process."""
		self.mu = mu * np.ones(size)
		self.theta = theta
		self.sigma = sigma
		
		random.seed(seed)
		self.size = size
		self.reset()  
		
		
	def reset(self):
		"""Reset the internal state (= noise) to mean (mu)."""
		self.state = copy.copy(self.mu)


	def sample(self):
		"""Update internal state and return it as a noise sample."""
		x = self.state
		dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.size)
		self.state = x + dx
		return self.state

		
		
class MADDPGAgent(object):
	"""Uses multiple DDPG agents which solves the environment by playing against each other"""
	
	def __init__(self, state_size, action_size, num_agents, seed):
		"""Initialize an MADDPGAgent object.
		
		Params :
			state_size (int)  --- dimension of each state
			action_size (int) --- dimension of each action
			num_agents		  --- no. of agents to train
			seed (int)		  --- random seed
		"""
		
		self.state_size = state_size
		self.action_size = action_size
		self.num_agents = num_agents
		self.seed = seed
		
		self.agents = [Agent(state_size, action_size, seed) for x in range(self.num_agents)]
		
		
	def act(self, states, add_noise = True):
		"""Initializes & returns actions for given states as per current policy (for each agent)
		
		Params:
			add_noise (boolean) --- introduce Ornstein-Uhlenbeck noise to the
									states in search for an optimal policy
		"""
		actions = np.zeros([self.num_agents, self.action_size])
		for index, agent in enumerate(self.agents):
			actions[index,:] = agent.act(states[index], add_noise)
		return actions
		
		
	def step(self, states, actions, rewards, next_states, dones):
		"""Sample experiences from a shared Replay buffer & update policy
		using the learned samples (per agent)
		
		Params (for each agent): 
			states		--- current states
			actions		--- actions taken for given states 
			rewards		--- feedbacks received
			next_states --- states generated based on the actions
			dones		--- Boolean values to see if episode finished
		"""
		
		sharedMem.add(states, actions, rewards, next_states, dones)		   
		for agent in self.agents:			  
			agent.step()
	
	
	def reset(self):
		"""Reset the noise when 'add_noise = True' is set (for each agent)"""
		for agent in self.agents:
			agent.reset()
			
			
	def save_weights(self):
		"""Save trained model weights (along with other hyperparameters)
		in .pth format for testing purposes & future use"""
		
		for i, agent in enumerate(self.agents):
			torch.save(agent.actor_local.state_dict(), 'agent{}_ckp_actor.pth'.format(i + 1))
			torch.save(agent.critic_local.state_dict(), 'agent{}_ckp_critic.pth'.format(i + 1))
			
			
	def load_weights(self):
		"""Load saved weights directly for testing purpose & use by
		third-parties for use without training a model"""
		
		if(not torch.cuda.is_available()):
			for i, agent in enumerate(self.agents):
				agent.actor_local.load_state_dict(torch.load('agent{}_ckp_actor.pth'.format(i + 1), map_location = torch.device('cpu')))
				agent.critic_local.load_state_dict(torch.load('agent{}_ckp_critic.pth'.format(i + 1), map_location = torch.device('cpu')))
		else:
			for i, agent in enumerate(self.agents):
				agent.actor_local.load_state_dict(torch.load('agent{}_ckp_actor.pth'.format(i + 1)))
				agent.critic_local.load_state_dict(torch.load('agent{}_ckp_critic.pth'.format(i + 1)))