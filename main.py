# Method to run the entire training process
from env import CryptoEnv
from agents.dqn import dqn
import pandas as pd
from data_utils.support_functions import load_ts
from agents.dqn import ReplayBuffer, Qnet
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from agents.dqn import train

learning_rate = 0.0005
gamma         = 0.98
buffer_limit  = 50000
batch_size    = 32

def main():
    ts = load_ts('close')
    # Step 1: Create and configure the environment
    env = CryptoEnv("YourEnvName")  # Replace "YourEnvName" with the name of your Gym environment
    q = Qnet()
    q_target = Qnet()
    q_target.load_state_dict(q.state_dict())
    memory = ReplayBuffer()
    print_interval = 20
    score = 0.0  
    optimizer = optim.Adam(q.parameters(), lr=learning_rate)

    for n_epi in range(10000):
        epsilon = max(0.01, 0.08 - 0.01*(n_epi/200)) #Linear annealing from 8% to 1%
        s = env.reset()
        done = False

        while not done:
            a = q.sample_action(torch.from_numpy(s).float(), epsilon)      
            s_prime, r, done, info = env.step(a)
            done_mask = 0.0 if done else 1.0
            memory.put((s,a,r,s_prime, done_mask))
            s = s_prime

            score += r
            if done:
                break

        if memory.size()>2000:
            train(q, q_target, memory, optimizer)

        if n_epi%print_interval==0 and n_epi!=0:
            q_target.load_state_dict(q.state_dict())
            print("n_episode :{}, score : {:.1f}, n_buffer : {}, eps : {:.1f}%".format(
                                                            n_epi, score/print_interval, memory.size(), epsilon*100))
            score = 0.0
        
    env.close()


if __name__ == '__main__':
    main()
