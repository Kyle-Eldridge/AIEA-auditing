from cnn import CNN
import cv2
from dqn import DQN
import gymnasium as gym
import os
import torch

env = gym.make("CarRacing-v3", continuous=False)
def preprocessor(x):
    return torch.from_numpy(cv2.cvtColor(x, cv2.COLOR_RGB2GRAY) / 255).float()
def reward_logger(episode, reward):
    print(f"Episode: {episode}, Reward: {reward}")
def get_learning_rate(step):
    step = min(step, 2000000)
    return 10 ** (-3 - step / 2000000)
def get_epsilon(step):
    step = min(step, 2000000)
    return 1 - (step / 2000000) * 0.99
nn = CNN((4, 96, 96), env.action_space.n)
dqn = DQN(env, nn, preprocessor=preprocessor, reward_logger=reward_logger, learning_rate=get_learning_rate, epsilon=get_epsilon, debug=True)
if os.path.exists("model.pth"):
    dqn.load()
dqn.learn(5000000)
