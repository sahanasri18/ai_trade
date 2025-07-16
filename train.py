import yfinance as yf
import numpy as np
from environment import TradingEnvironment
from agent import DQNAgent
from data_utils import preprocess_data

# Load data
symbol = "AAPL"
df = yf.download(symbol, start="2020-01-01", end="2023-01-01")
df = preprocess_data(df)

# Create Environment and Agent
env = TradingEnvironment(df)

state_size = env.observation_space.shape[0]
action_size = env.action_space.n
agent = DQNAgent(state_size, action_size)

episodes = 50

for e in range(episodes):
    state = env.reset()
    total_reward = 0

    for time in range(len(df) - 1):
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

        if done:
            print(f"✅ Episode: {e+1}/{episodes}, Total Reward: {total_reward:.2f}, Net Worth: {env.net_worth:.2f}")
            break

    agent.replay()

# Optional: Save the model
agent.model.save("trained_trading_model.h5")
