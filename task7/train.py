from cnn import CNN
import cv2
from a3c import A3C
import gymnasium as gym
import torch

env_supplier = lambda: gym.make("CarRacing-v3", continuous=False)
env = env_supplier()
def preprocessor(x):
    return torch.from_numpy(cv2.cvtColor(x, cv2.COLOR_RGB2GRAY) / 255).float()
nn = CNN((4, 96, 96), env.action_space.n)
a3c = A3C(env_supplier, nn, preprocessor=preprocessor, reward_logger=lambda reward: print(f'Reward: {reward}'), debug=True)
a3c.learn(5000000)
print()
