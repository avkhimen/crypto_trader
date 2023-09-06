import numpy as np
import pandas as pd

class CryptoEnv()
    def __init__(self, all_price_series):
        self.all_price_series = all_price_series
    def reset(self):
        self.price_size = self.all_price_series.shape[0
        random_index = np.random.choice(self.price_size, 1)
        pass
    def step(self):
        pass