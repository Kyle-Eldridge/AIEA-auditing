import copy
import torch
import torch.optim as optim
import gymnasium as gym
import numpy as np
import random

class DQN:
    def __init__(self, env, nn, learning_rate=1e-4, gamma=0.98, memory_size=20000, batch_size=64, preprocessor=None, reward_logger=None):
        self.env = env
        self.nn = nn
        self.target_nn = copy.deepcopy(nn)
        self.gamma = gamma
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.preprocessor = preprocessor if preprocessor is not None else lambda x: x
        self.reward_logger = reward_logger
        self.action_space = env.action_space.n
        self.replay_memory = []
        self.replay_index = 0
        self.optimizer = optim.Adam(nn.parameters(), lr=learning_rate)
    
    def select_action(self, obs, epsilon):
        if random.random() < epsilon:
            return int(random.random() * self.action_space)
        with torch.no_grad():
            return self.nn(obs.unsqueeze(0)).argmax(dim=1).item()


    def gradient_step(self):
        if len(self.replay_memory) < self.batch_size * 2:
            return
        indexes = set()
        memory_size = len(self.replay_memory)
        while len(indexes) < self.batch_size:
            index = random.randint(0, memory_size-1)
            if (index - self.replay_index) % memory_size < 4:
                continue
            if self.replay_memory[index][4] != self.replay_memory[(index - 4) % memory_size][4]:
                continue
            indexes.add(index)

        batch = [self.replay_memory[i] for i in indexes]
        states = torch.stack([torch.stack([self.replay_memory[(i - j - 1) % memory_size][0] for j in range(4)]) for i in indexes])
        next_states = torch.stack([torch.stack([self.replay_memory[(i - j) % memory_size][0] for j in range(4)]) for i in indexes])
        actions = torch.tensor([b[1] for b in batch], dtype=torch.long)
        rewards = torch.tensor([b[2] for b in batch], dtype=torch.float32)
        done = torch.tensor([b[3] for b in batch], dtype=torch.bool)
        
        q_values = self.nn(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        self.optimizer.zero_grad()

        with torch.no_grad():
            targets = rewards + self.gamma * self.target_nn(next_states).max(dim=1).values * ~done
        loss = torch.nn.functional.smooth_l1_loss(q_values, targets)
        loss.backward()
        self.optimizer.step()

    def learn(self, timesteps):
        try:
            steps = 0
            episode = 0
            total_reward = 0
            obs, info = self.env.reset()
            obs = self.preprocessor(obs)
            sequence = [obs] * 4
            while steps < timesteps:
                epsilon = 1-steps/timesteps*0.99
                current = torch.stack(sequence)
                action = self.select_action(current, epsilon)
                obs, reward, terminated, truncated, info = self.env.step(action)
                total_reward += reward
                obs = self.preprocessor(obs)
                sequence.pop(0)
                sequence.append(obs)
                
                if len(self.replay_memory) < self.memory_size:
                    self.replay_memory.append((sequence[3], action, reward, terminated, episode))
                else:
                    self.replay_memory[self.replay_index] = (sequence[3], action, reward, terminated, episode)
                self.replay_index = (self.replay_index + 1) % self.memory_size
                
                self.gradient_step()
                if steps % 1000 == 0:
                    self.target_nn.load_state_dict(self.nn.state_dict())
                
                if terminated or truncated:
                    if self.reward_logger is not None:
                        self.reward_logger(total_reward)
                    total_reward = 0
                    obs, info = self.env.reset()
                    obs = self.preprocessor(obs)
                    sequence = [obs] * 4
                    episode += 1
                steps += 1
                if steps % (timesteps // 100) == 0:
                    print(f"{steps / timesteps * 100:.2f}%")
        except KeyboardInterrupt:
            pass

from cnn import CNN
import cv2
env = gym.make("CarRacing-v3", continuous=False)
preprocessor = lambda x: torch.from_numpy(cv2.cvtColor(x, cv2.COLOR_RGB2GRAY) / 255).float()
nn = CNN((4, 96, 96), env.action_space.n)
dqn = DQN(env, nn, preprocessor=preprocessor, reward_logger=lambda reward: print(reward))
dqn.learn(10000)
