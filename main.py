import sys

sys.path.append('agents/')

# Method to run the entire training process
from env import CryptoEnv
import pandas as pd
from data_utils.support_functions import load_ts
from dqn import ReplayBuffer, train
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np

learning_rate = 0.0005
gamma         = 0.98
buffer_limit  = 50000
batch_size    = 32

class Qnet(nn.Module):
    def __init__(self):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(37, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
      
    def sample_action(self, obs, epsilon):
        out = self.forward(obs)
        coin = random.random()
        if coin < epsilon:
            return random.randint(0,1)
        else : 
            return out.argmax().item()

def main():
    ts = load_ts('close_price')
    # Step 1: Create and configure the environment
    env = CryptoEnv(ts, lookup_interval=36, window_size=72)
    q = Qnet()
    q_target = Qnet()
    q_target.load_state_dict(q.state_dict())
    memory = ReplayBuffer()
    print_interval = 20
    score = 0.0  
    optimizer = optim.Adam(q.parameters(), lr=learning_rate)

    for n_epi in range(1000000):
        epsilon = max(0.05, 0.05 - 0.05*(n_epi/2000)) #Linear annealing from 8% to 1%
        s = env.reset()
        #print('reset state')
        done = False

        while not done:
            a = q.sample_action(torch.from_numpy(np.array(s)).float(), epsilon)      
            s_prime, r, done, info = env.step(s, a)
            done_mask = 0.0 if done else 1.0
            memory.put((s,a,r,s_prime, done_mask))
            s = s_prime

            score += r
            if done:
                #print('Episode completed')
                break

        if memory.size()>2000:
            train(q, q_target, memory, optimizer)

        if n_epi%print_interval==0 and n_epi!=0:
            q_target.load_state_dict(q.state_dict())
            print("n_episode :{}, score : {:.5f}, n_buffer : {}, eps : {:.1f}%".format(
                                                            n_epi, score/print_interval, memory.size(), epsilon*100))
            score = 0.0
        
    env.close()


if __name__ == '__main__':
    main()
