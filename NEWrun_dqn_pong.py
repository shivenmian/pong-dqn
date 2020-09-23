from Wrapper.layers import *
from Wrapper.wrappers import make_atari, wrap_deepmind, wrap_pytorch
import math, random
import gym
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd 
import torch.nn.functional as F
USE_CUDA = torch.cuda.is_available()
from NEWdqn import QLearner, compute_td_loss, ReplayBuffer
'''
def plots(frame_idx, rewards, losses):
    plt.figure(figsize=(20, 5))
    plt.subplot(131)
    plt.title('Frame %s. Last-10 reward: %s' % (frame_idx, np.mean(rewards[-10:], 0)[1]))
    plt.plot(rewards)
    plt.subplot(132)
    plt.title('loss')
    plt.plot(losses)
    plt.savefig("plots/plot-frame-{}.png".format(frame_idx))
'''
env_id = "PongNoFrameskip-v4"
env = make_atari(env_id)
env = wrap_deepmind(env)
env = wrap_pytorch(env)

num_frames = 3000000
batch_size = 32
gamma = 0.99
record_idx = 10000

replay_initial = 10000
replay_buffer = ReplayBuffer(100000)
model = QLearner(env, num_frames, batch_size, gamma, replay_buffer)
#model.load_state_dict(torch.load("models/model-2900000.pth"))
# New code for second model
target_model = QLearner(env, num_frames, batch_size, gamma, replay_buffer)
target_model.copy_from(model)

optimizer = optim.Adam(model.parameters(), lr=0.00001)
if USE_CUDA:
    model = model.cuda()
# New code     
    target_model = target_model.cuda()
    print("Using cuda")

# Epsilon is the chance of making a random move
# It trades of exploration vs exploitation
epsilon_start = 1.0
epsilon_final = 0.01
epsilon_decay = 30000
epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * frame_idx / epsilon_decay)

losses = []
all_rewards = []
episode_reward = 0

state = env.reset()

for frame_idx in range(1, num_frames + 1):
    #print("Frame: " + str(frame_idx))

    epsilon = epsilon_by_frame(frame_idx)
    # Perform an action based on current state and epsilon
    action = model.act(state, epsilon)
    
    next_state, reward, done, _ = env.step(action)
    # Push <s, a, r, s'> t the replay buffer
    replay_buffer.push(state, action, reward, next_state, done)

    # store the new state and the cummulative reward for the game
    state = next_state
    episode_reward += reward

    # game is over
    if done:
        state = env.reset()
        all_rewards.append((frame_idx, episode_reward))
        episode_reward = 0

    # replay buffer is big enough
    if len(replay_buffer) > replay_initial:
        #New Code
        loss = compute_td_loss(model, target_model, batch_size, gamma, replay_buffer)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append((frame_idx, loss.data.cpu().numpy()[()]))
        #print(loss.data, type(loss.data.cpu().numpy()), loss.data.cpu().numpy().shape, loss.data.cpu().numpy(), loss.data.cpu().numpy().tolist())

    if frame_idx % 10000 == 0 and len(replay_buffer) <= replay_initial:
        print('#Frame: %d, preparing replay buffer' % frame_idx)

    if frame_idx % 10000 == 0 and len(replay_buffer) > replay_initial:
        print('#Frame: %d, Loss: %f' % (frame_idx, np.mean(losses, 0)[1]))
        print('Last-10 average reward: %f' % np.mean(all_rewards[-10:], 0)[1])

    # New code
    if frame_idx % 50000 == 0:
        target_model.copy_from(model)

    if frame_idx >= 2000000 and frame_idx % 10000 == 0:
        torch.save(model.state_dict(), "models/model-{}.pth".format(frame_idx))
        #torch.save(target_model.state_dict(), "models/targetmodel-{}.pth".format(frame_idx))
        print("Model saved at frame {}".format(frame_idx))

np.save("rewards", all_rewards)
np.save("losses", losses)
