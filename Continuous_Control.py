from unityagents import UnityEnvironment
import numpy as np
from ddpg_agent import Agent
import random
import torch
from collections import deque
import matplotlib.pyplot as plt

env = UnityEnvironment(file_name='Reacher_Windows_x86_64/Reacher.exe',no_graphics=True)

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=False)[brain_name]

# number of agents
num_agents = len(env_info.agents)

# size of each action
action_size = brain.vector_action_space_size

# size of each state
states = env_info.vector_observations
state_size = states.shape[1]

# init agent
agent = Agent(state_size, action_size, random_seed=2)

def ddpg(n_episodes=500, target_score=30.0, print_every=1, window_len=100, updateAfterEvery=20, numUpdateCycles=10):
    scoresOverEpisodes = []
    scores_deque = deque(maxlen=window_len)    
    actions = np.zeros([num_agents, action_size])

    for i_episode in range(1, n_episodes+1):
    
        env_info = env.reset(train_mode=True)[brain_name]        
        states = env_info.vector_observations
        agent.reset()
        scores = np.zeros(num_agents)                          # initialize the score (for each agent)
        t = 0

        #for t in range(max_t):
        while(True):
            t += 1
            actions = agent.act(states)                        # get action from policy            
            env_info = env.step(actions)[brain_name]           # send all actions to the environment
            next_states = env_info.vector_observations         # get next state (for each agent)
            rewards = env_info.rewards                         # get reward (for each agent)
            dones = env_info.local_done                        # see if episode finished

            for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
                agent.store(state, action, reward, next_state, done)

            if t % updateAfterEvery == 0:
                for i in range(numUpdateCycles):
                    agent.learnFromBufferSamples()

            scores += rewards                                  # update the score (for each agent)
            states = next_states                               # roll over states to next time step

            if np.any(dones):                                  # exit loop if episode finished
                break
   
        scoresOverEpisodes.append(np.mean(scores))
        scores_deque.append(np.mean(scores))

        if i_episode % print_every == 0:
            print('\rEpisode {}\tCurrent mean score: {:.2f}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores),np.mean(np.mean(scores_deque))))      
    
        if np.mean(scores_deque) >= target_score:
            # solved
            break

    torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
    torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')

    return scoresOverEpisodes

scores = ddpg()

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(scores)+1), scores)
plt.ylabel('Avg. Score')
plt.xlabel('Episode #')
plt.show()
fig.savefig('scores.png')

env.close()