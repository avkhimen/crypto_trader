import numpy as np

class CryptoEnv():
    def __init__(self, all_price_series, lookup_interval, window_size):
        self.all_price_series = all_price_series
        self.window_size = window_size
        self.lookup_interval = lookup_interval
    def reset(self):
        self.step_order = 0
        cash_state = [0] # all cash - 0, all btc - 1
        cash_state.extend(self.select_random_window())
        cash_state.extend([0])
        return cash_state
    def select_random_window(self):
        """Select a window from the array with a random starting index."""
        if self.window_size > len(self.all_price_series):
            raise Exception("Please choose a larger window size")
        # Choose a random starting index
        self.start_index = np.random.randint(0, len(self.all_price_series)-1000)
        self.ser = np.array(self.all_price_series[self.start_index:self.start_index+self.window_size+self.lookup_interval+250])
        self.first_elem = self.ser[0]
        self.ser_normalized = (np.array(self.ser)/self.first_elem).tolist()
        return self.ser_normalized[:self.lookup_interval]
    def step(self, state, action):
        cash_state = state[0]
        if cash_state == 0:
            if action == 0: # do nothing
                reward = 0
                next_cash_state = 0
            elif action == 1: # buy
                reward = self.ser_normalized[self.lookup_interval + self.step_order+1] - self.ser_normalized[self.lookup_interval + self.step_order]
                next_cash_state = 1
            elif action == 2: # sell
                reward = 0
                next_cash_state = 0
        elif cash_state == 1:
            if action == 0: # do nothing
                reward = self.ser_normalized[self.lookup_interval + self.step_order+1] - self.ser_normalized[self.lookup_interval + self.step_order]
                next_cash_state = 1
            elif action == 1: # buy
                reward = 0
                next_cash_state = 1
            elif action == 2: # sell
                reward = min(0, -(self.ser_normalized[self.lookup_interval + self.step_order+1] - self.ser_normalized[self.lookup_interval + self.step_order]))
                next_cash_state = 0
        next_state = [next_cash_state] # must be a list
        self.step_order += 1
        self.ser = np.array(self.all_price_series[self.start_index+self.step_order:self.start_index+self.lookup_interval+self.step_order+250])
        self.ser_normalized = (np.array(self.ser)/self.first_elem).tolist()
        next_state.extend(self.ser_normalized[:self.lookup_interval])
        next_state.extend([action])
        done = False
        if self.step_order == self.window_size:
            done = True
        info = {}
        return next_state, reward, done, info