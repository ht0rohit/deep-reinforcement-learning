# Project 2: Continuous Control

## Introduction

For this project, I've worked with the Reacher environment to train a robotic arm to reach target locations.

The image is a reference to how in the actual environment with multiple agents, solve that environment. This repository is an implementation of the same with a single agent.

![Trained Agent0](https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif)

![Trained Agent1](https://github.com/ht0rohit/Deep-Reinforcement-Learning/blob/master/1ContinuousControl/Assets/trained.gif)

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

#### Solving the Environment

The task is episodic, and in order to solve the environment, an agent must get an average score of atleast +30 over 100 consecutive episodes.

## Getting Started

  1. Download the environment from one of the links below. You need to only select the environment that matches your operating system:
  
      - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
      - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
      - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)
      - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)
      
      (*For Windows users*) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of     the Windows operating system.

      (*For AWS*) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

  2. Place the file in the GitHub repository, in the 1ContinuousControl/ folder, and unzip (or decompress) the file.
  
## Instructions

Follow the instructions in `Continuous_Control.ipynb` to get started with training the agent!
