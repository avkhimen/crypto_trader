# crypto_trader
Repo for RL-based (and maybe others) trading agents

The main approach is to break the continuing bitcoin OHLC price data into episodes of fixed length. Once the episodes are made, each episode is normalized by dividing it by the starting price (one of OHLC prices) in the series. Dividing by the opening price will scale the time series, which will ensure all episodes can be used with the same policy. Then a Deep RL-based policy will be trained using one of SAC, TD3, and PPO algorithms. The action is the percentage (0-1) of the portfolio that is being held in the asset in question.
