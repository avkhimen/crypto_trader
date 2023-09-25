# Method to run the entire training process
from env import CryptoEnv
from agents.dqn import dqn
import pandas as pd
from data_utils.support_functions import load_ts

def main():
    ts = load_ts('close')
    # Step 1: Create and configure the environment
    env = CryptoEnv("YourEnvName")  # Replace "YourEnvName" with the name of your Gym environment
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # Step 2: Initialize the SAC agent
    agent = SAC(state_dim, action_dim,  # Specify state and action dimensions
                actor_lr=0.001, critic_lr=0.001, alpha_lr=0.001)  # Tune hyperparameters

    # Step 3: Training loop
    max_episodes = 1000  # Adjust as needed
    for episode in range(max_episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            # Step 4: Select an action from the agent's policy
            action = agent.select_action(state)

            # Step 5: Take a step in the environment
            next_state, reward, done, _ = env.step(action)

            # Step 6: Store the transition in the replay buffer
            agent.replay_buffer.add(state, action, reward, next_state, done)

            # Step 7: Train the agent
            agent.train()

            # Update the current state and total reward
            state = next_state
            total_reward += reward

        print(f"Episode {episode + 1} - Total Reward: {total_reward}")

    # Step 8: Save the trained model if needed
    agent.save_model("sac_model.pth")

if __name__ == '__main__':
    main()