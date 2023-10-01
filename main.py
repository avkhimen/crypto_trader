import sys

sys.path.append('agents/')

# Method to run the entire training process
from env import CryptoEnv
from data_utils.support_functions import load_ts
from dqn import train
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np
import collections

learning_rate = 0.01
gamma         = 0.99
buffer_limit  = 200000
batch_size    = 64

class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)
    
    def put(self, transition):
        self.buffer.append(transition)
    
    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []
        
        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])

        return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
               torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
               torch.tensor(done_mask_lst)
    
    def size(self):
        return len(self.buffer)

class Qnet(nn.Module):
    def __init__(self):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(26, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
      
    def sample_action(self, obs, epsilon):
        out = self.forward(obs)
        coin = random.random()
        if coin < epsilon:
            return random.randint(0,2)
        else : 
            return out.argmax().item()

def main():
    ts = load_ts('close_price')
    env = CryptoEnv(ts, lookup_interval=24, window_size=48)
    q = Qnet()
    q_target = Qnet()
    q_target.load_state_dict(q.state_dict())
    memory = ReplayBuffer()
    print_interval = 20
    score = 0.0  
    optimizer = optim.Adam(q.parameters(), lr=learning_rate)

    for n_epi in range(1000000):
        epsilon = 0.05
        s = env.reset()
        done = False

        while not done:
            a = q.sample_action(torch.from_numpy(np.array(s)).float(), epsilon)      
            s_prime, r, done, info = env.step(s, a)
            done_mask = 0.0 if done else 1.0
            memory.put((s,a,10*r,s_prime, done_mask))
            s = s_prime

            score += 10*r
            if done:
                break
        with open('scores.txt', 'a') as f:
            f.write(f"\n {n_epi} {score}")

        if memory.size()>20000:
            train(q, q_target, memory, optimizer)

        if n_epi%print_interval==0 and n_epi!=0:
            q_target.load_state_dict(q.state_dict())
            print("n_episode :{}, score : {:.5f}, n_buffer : {}, eps : {:.1f}%".format(
                                                            n_epi, score, memory.size(), epsilon*100))
            score = 0.0
        
    env.close()


if __name__ == '__main__':
    main()
