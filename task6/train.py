from cnn import CNN
import cv2
from dqn import DQN
import gymnasium as gym
import torch

env = gym.make("CarRacing-v3", continuous=False)
def preprocessor(x):
    return torch.from_numpy(cv2.cvtColor(x, cv2.COLOR_RGB2GRAY) / 255).float()
nn = CNN((4, 96, 96), env.action_space.n)
dqn = DQN(env, nn, preprocessor=preprocessor, reward_logger=lambda reward: print(reward))
dqn.learn(5000000)
