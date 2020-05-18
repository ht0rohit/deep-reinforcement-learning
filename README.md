# Deep-Reinforcement-Learning

## Introduction
The repository contains various projects I worked upon during learning Deep Reinforcement Learning (Deep RL) using the environments - OpenAI Gym &amp; Unity ML-agents

The projects are in the order of advancements in the algorithms used in Deep RL. I'll soon be sharing more implementations of some of the most popular environments in OpenAI Gym using these algorithms.

## Getting Started

### Dependencies
To set up your python environment to run the code in this repository, follow the instructions below:

  1. Create (and activate) a new environment with Python 3.6.
  
      - **Linux** or **Mac**:
     
            conda create --name deeprl python=3.6
            source activate deeprl
      
      - **Windows**:
      
            conda create --name deeprl python=3.6 
            activate deeprl
  
  2. Follow the instructions in [this](https://github.com/openai/gym) repository to perform a minimal install of OpenAI gym & then            install some other environments as per your need.
  
  3. Install PyTorch  
  
      You can use the code below to install PyTorch without CUDA (for Windows) or refer to [here](https://pytorch.org/).
  
          pip install torch==1.5.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
  
  4. Clone the repository and navigate to the python/ folder. Then, install several dependencies.
    
          git clone https://github.com/ht0rohit/Deep-Reinforcement-Learning.git
          cd Deep-Reinforcement-Learning/python
          pip install .
  
  5. Create an IPython kernel for the deeprl environment.
    
          python -m ipykernel install --user --name deeprl --display-name "deeprl"
  
  6. Before running code in a notebook, change the kernel to match the deeprl environment by using the drop-down Kernel menu.
