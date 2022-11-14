# Project 2: Continuous Control


## Introduction

The purpose of this script is to train a Reinforcement Learning agent for the Reacher environment, which includes twenty parallel working identical double-jointed arms to move towards target locations.

The tasks in this environment are episodic. 
Rewards of +0.1 are given for an agent being the the goal location.
To do so, the robots may be input by a four dimensional vector with continuously ranging between $[-1,1]$. 
The states of each agent is represented by a 33 dimensional vector, coding positions, rotation, velocity and angular veocity if the arm.

The training is carried out in a distributed manner by means of [deep deterministic policy gradient (DDPG)](https://arxiv.org/abs/1509.02971).


## Getting Started

1. Set up a Python 3.6 environment including the following packages:
    - `Torch 0.4.0`
    - `unityagents 0.4.0`
2. Download the environment
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)     
3. Place the file into the root folder, and unzip (or decompress) it.


## Instructions

Executing `Continuous_Control.py` will start the training of the agent. 
It will output a plot of the averaged score over 100 consecutive time frames and all 20 agents. 
Resulting weights of the trained actor and critic will be stored in the files `checkpoint_*.pth`.

## Sources

The implementation builds up on a code framework provided by [Udacity's Reinforcement Learning Exprert Nano degree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893).
It also uses the `Reacher`-environment of [Unity](https://unity.com/de/products/machine-learning-agents).
The implemented actor-critic reinforcement learning method implemented is [deep deterministic policy gradient (DDPG)](https://arxiv.org/abs/1509.02971).