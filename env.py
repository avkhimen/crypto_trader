import numpy as np
import pandas as pd

class CryptoEnv():
    def __init__(self, all_price_series, lookup_interval=1, window_size=2):
        self.all_price_series = all_price_series
        self.window_size = window_size
        self.lookup_interval = lookup_interval
    def reset(self):
        self.step_order = 0
        cash_state = [0] # all cash - 0, all btc - 1
        cash_state.extend(self.select_random_window())
        return cash_state
    def select_random_window(self):
        """Select a window from the array with a random starting index."""
        if self.window_size > len(self.all_price_series):
            return []
        # Choose a random starting index
        start_index = np.random.randint(0, len(self.all_price_series) - self.window_size + 1)
        self.ser = np.array(self.all_price_series[start_index - self.lookup_interval:start_index + self.window_size])
        self.ser = self.normalize_price_series()
        
        return self.ser.tolist()
    def normalize_price_series(self):
        return self.ser/self.ser[0]
    def step(self, state, action):
        cash_state = state[0]
        if cash_state == 0:
            if action == 0: # do nothing
                reward = 0
                next_cash_state = 0
            elif action == 1: # buy
                reward = self.ser[self.step_order + 1] -  self.ser[self.step_order]
                next_cash_state = 1
            elif action == 2: # sell
                reward = 0
                next_cash_state = 0
        elif cash_state == 1:
            if action == 0: # do nothing
                reward = self.ser[self.step_order + 1] -  self.ser[self.step_order]
                next_cash_state = 1
            elif action == 1: # buy
                reward = 0
                next_cash_state = 1
            elif action == 2: # sell
                if self.ser[self.step_order + 1] -  self.ser[self.step_order] < 0:
                    reward = 0
                else:
                    reward = -(self.ser[self.step_order + 1] -  self.ser[self.step_order])
                next_cash_state = 0
        next_state = [next_cash_state] # must be a list
        next_state.extend()
        done = False
        if self.step_order == self.window_size:
            done = True
        info = {}
        self.step_order += 1
        return next_state, reward, done, info