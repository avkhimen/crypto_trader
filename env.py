import numpy as np
import pandas as pd

class CryptoEnv():
    def __init__(self, all_price_series, window_size=24):
        self.all_price_series = all_price_series
        self.window_size = window_size
    def reset(self):
        return self.select_random_window()
    def select_random_window(self):
        """Select a window from the array with a random starting index."""
        if self.window_size > len(self.all_price_series):
            return []
        # Choose a random starting index
        start_index = np.random.randint(0, len(self.all_price_series) - self.window_size + 1)
        self.ser = np.array(self.all_price_series[start_index:start_index + self.window_size])
        self.ser = self.normalize_price_series()
        return self.ser
    def normalize_price_series(self):
        return self.ser/self.ser[0]
    # def step(self, action):
    #     if action == ''
    #     next_state = []
    #     reward = 0
    #     done = False
    #     info = {}
    #     return next_state, reward, done, info