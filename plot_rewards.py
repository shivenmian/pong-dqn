import numpy as np
import matplotlib.pyplot as plt

def plot_rewards_lr():
    rewards_norm = np.load("rewards_normal.npy")
    rewards_big = np.load("lrbig/rewards_lrbig.npy")
    rewards_small = np.load("lrsmall/rewards_lrsmall.npy")
    rewards_bigger = np.load("lrbigger/rewards_lrbigger.npy")

    frames = [i for i in range(10000, 3000001, 10000)]

    last10_norm = [np.mean(rewards_norm[rewards_norm[:,0] <= i][-10:], 0)[1] for i in frames]
    last10_big = [np.mean(rewards_big[rewards_big[:,0] <= i][-10:], 0)[1] for i in frames]
    last10_small = [np.mean(rewards_small[rewards_small[:,0] <= i][-10:], 0)[1] for i in frames]
    last10_bigger = [np.mean(rewards_bigger[rewards_bigger[:,0] <= i][-10:], 0)[1] for i in frames]

    print('norm0.00001', np.max(last10_norm), 'big0.00005', np.max(last10_big), 'small0.000005', np.max(last10_small), 'bigger0.0005', np.max(last10_bigger))
    plt.figure(figsize=(30,5))
    plt.plot(frames, last10_bigger, label='lr=0.0005')
    plt.plot(frames, last10_big, label='lr=0.00005')
    plt.plot(frames, last10_norm, label='lr=0.00001')
    plt.plot(frames, last10_small, label='lr=0.000005')
    plt.legend()
    plt.title("Variation of Reward with LR")
    plt.xlabel("Number of frames (millions)")
    plt.ylabel("Last-10 episode Cumulative Reward")
    plt.show()

    #plt.plot(rewards_norm[:,0], rewards_norm[:,1])

def plot_losses_lr():
    losses_norm = np.load("losses_normal.npy")
    losses_big = np.load("lrbig/losses_lrbig.npy")
    losses_small = np.load("lrsmall/losses_lrsmall.npy")
    losses_bigger = np.load("lrbigger/losses_lrbigger.npy")
    #frames = [i for i in range(10000, 3000001, 10000)]

    #last10_norm = [np.mean(rewards_norm[rewards_norm[:,0] <= i][-10:], 0)[1] for i in frames]
    #last10_big = [np.mean(rewards_big[rewards_big[:,0] <= i][-10:], 0)[1] for i in frames]
    #last10_small = [np.mean(rewards_small[rewards_small[:,0] <= i][-10:], 0)[1] for i in frames]

    plt.figure(figsize=(30,5))
    # plt.plot(frames, last10_norm, label='lr=0.00001')
    # plt.plot(frames, last10_big, label='lr=0.00005')
    # plt.plot(frames, last10_small, label='lr=0.000005')
    print("norm0.00001 ({}, {}), big0.00005 ({}, {}), small0.000005 ({}, {}), bigger0.0005 ({}, {})".format(np.max(losses_norm[:,1]), np.min(losses_norm[:,1]), 
        np.max(losses_big[:,1]), np.min(losses_big[:,1]), np.max(losses_small[:,1]), np.min(losses_small[:,1]), np.max(losses_bigger[:,1]), np.min(losses_bigger[:,1])))
        
    plt.plot(losses_bigger[:,0], losses_bigger[:,1], label='lr=0.0005')
    plt.plot(losses_big[:,0], losses_big[:,1], label='lr=0.00005')
    plt.plot(losses_norm[:,0], losses_norm[:,1], label='lr=0.00001')
    plt.plot(losses_small[:,0], losses_small[:,1], label='lr=0.000005')
    plt.title("Variation of Loss with LR")
    plt.xlabel("Number of frames (millions)")
    plt.ylabel("TD Loss")
    plt.legend()
    plt.show()

    #plt.plot(rewards_norm[:,0], rewards_norm[:,1])

def plot_rewards_gamma():
    rewards_norm = np.load("rewards_normal.npy")
    rewards_big = np.load("gammabig/rewards_gammabig.npy")
    rewards_small = np.load("gammasmall/rewards_gammasmall.npy")
    rewards_mid = np.load("gammamid/rewards_gammamid.npy")

    frames = [i for i in range(10000, 3000001, 10000)]

    last10_norm = [np.mean(rewards_norm[rewards_norm[:,0] <= i][-10:], 0)[1] for i in frames]
    last10_big = [np.mean(rewards_big[rewards_big[:,0] <= i][-10:], 0)[1] for i in frames]
    last10_small = [np.mean(rewards_small[rewards_small[:,0] <= i][-10:], 0)[1] for i in frames]
    last10_mid = [np.mean(rewards_mid[rewards_mid[:,0] <= i][-10:], 0)[1] for i in frames]

    print('norm0.99', np.max(last10_norm), 'big0.75', np.max(last10_big), 'small0.25', np.max(last10_small), 'mid0.5', np.max(last10_mid))
    plt.figure(figsize=(30,5))
    plt.plot(frames, last10_norm, label='gamma=0.99')
    plt.plot(frames, last10_big, label='gamma=0.75')
    plt.plot(frames, last10_mid, label='gamma=0.5')
    plt.plot(frames, last10_small, label='gamma=0.25')
    plt.legend()
    plt.title("Variation of Reward with Gamma")
    plt.xlabel("Number of frames (millions)")
    plt.ylabel("Last-10 episode Cumulative Reward")
    plt.show()

    #plt.plot(rewards_norm[:,0], rewards_norm[:,1])

def plot_losses_gamma():
    losses_norm = np.load("losses_normal.npy")
    losses_big = np.load("gammabig/losses_gammabig.npy")
    losses_small = np.load("gammasmall/losses_gammasmall.npy")
    losses_mid = np.load("gammamid/losses_gammamid.npy")
    #frames = [i for i in range(10000, 3000001, 10000)]

    #last10_norm = [np.mean(rewards_norm[rewards_norm[:,0] <= i][-10:], 0)[1] for i in frames]
    #last10_big = [np.mean(rewards_big[rewards_big[:,0] <= i][-10:], 0)[1] for i in frames]
    #last10_small = [np.mean(rewards_small[rewards_small[:,0] <= i][-10:], 0)[1] for i in frames]

    plt.figure(figsize=(30,5))
    # plt.plot(frames, last10_norm, label='lr=0.00001')
    # plt.plot(frames, last10_big, label='lr=0.00005')
    # plt.plot(frames, last10_small, label='lr=0.000005')
    print("norm0.99 ({}, {}), big0.75 ({}, {}), mid0.5 ({}, {}), small0.25 ({}, {})".format(np.max(losses_norm[:,1]), np.min(losses_norm[:,1]), 
        np.max(losses_big[:,1]), np.min(losses_big[:,1]), np.max(losses_mid[:,1]), np.min(losses_mid[:,1]), np.max(losses_small[:,1]), np.min(losses_small[:,1])))
    
    plt.plot(losses_norm[:,0], losses_norm[:,1], label='gamma=0.99')
    plt.plot(losses_big[:,0], losses_big[:,1], label='gamma=0.75')
    plt.plot(losses_mid[:,0], losses_mid[:,1], label='gamma=0.5')
    plt.plot(losses_small[:,0], losses_small[:,1], label='gamma=0.25')
    plt.title("Variation of Loss with Gamma")
    plt.xlabel("Number of frames (millions)")
    plt.ylabel("TD Loss")
    plt.legend()
    plt.show()

    #plt.plot(rewards_norm[:,0], rewards_norm[:,1])

#plot_rewards()
plot_rewards_lr()