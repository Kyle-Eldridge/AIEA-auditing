import triton # Fix import order bug
import gymnasium as gym
from stable_baselines3 import PPO

env = gym.make("CarRacing-v3", render_mode="rgb_array")
model = PPO("CnnPolicy", env, tensorboard_log="log/", verbose=1,
            gamma=0.99, gae_lambda=0.95, batch_size=128, n_steps=512, learning_rate=1e-4)

model.learn(4e6)

