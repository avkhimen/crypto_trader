import numpy as np
import pandas as pd

class CryptoEnv():
    def __init__(self, all_price_series, window_size=24):
        self.all_price_series = all_price_series
        self.window_size = window_size
    def reset(self):
        s = 
        return s
    def step(self, action):
        if action == ''
        next_state = []
        reward = 0
        done = False
        info = {}
        return next_state, reward, done, info