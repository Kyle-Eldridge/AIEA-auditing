import copy
import torch
from threading import Thread, Lock
import torch.optim as optim
import gymnasium as gym
import numpy as np
import os
import random

torch.set_num_threads(4)
torch.set_num_interop_threads(1)

class DQN:
    def __init__(self, env, nn, learning_rate=1e-4, epsilon=0.1, gamma=0.98, memory_size=20000, n_step=5, batch_size=64, preprocessor=None, reward_logger=None, fileName="model.pth", debug=False):
        self.env = env
        self.test_env = copy.deepcopy(env)
        self.nn = nn
        self.target_nn = copy.deepcopy(nn)
        self.gamma = gamma
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.preprocessor = preprocessor if preprocessor is not None else lambda x: x
        floatType = type(1.23)
        self.learning_rate = (lambda _: learning_rate) if type(learning_rate) == floatType else learning_rate
        self.epsilon = (lambda _: epsilon) if type(epsilon) == floatType else epsilon
        self.n_step = n_step
        self.reward_logger = reward_logger
        self.fileName = fileName
        self.action_space = env.action_space.n
        self.replay_memory = []
        self.replay_index = 0
        self.steps = 0
        self.episode = 0
        self.best_reward = -1e9
        self.optimizer = optim.Adam(nn.parameters(), lr=self.learning_rate(0))
        self.nn_lock = Lock()
        self.best_lock = Lock()
        self.interrupt = False
        self.debug = debug
    
    def select_action(self, obs, epsilon):
        if random.random() < epsilon:
            return int(random.random() * self.action_space)
        with torch.no_grad():
            return self.nn(obs.unsqueeze(0)).argmax(dim=1).item()

    def gradient_step(self):
        if len(self.replay_memory) < self.batch_size * 2.5:
            return
        indexes = set()
        memory_size = len(self.replay_memory)
        while len(indexes) < self.batch_size:
            index = random.randint(0, memory_size-1)
            if (index - self.replay_index) % memory_size < 4:
                continue
            if self.replay_memory[index][4] != self.replay_memory[(index - 4) % memory_size][4]:
                continue
            if (self.replay_index - index) % memory_size < self.n_step:
                continue
            indexes.add(index)
        indexes = list(indexes)

        n_steps = []
        for i in indexes:
            added = False
            for j in range(1, self.n_step):
                if self.replay_memory[(i + j) % memory_size][4] != self.replay_memory[i][4]:
                    n_steps.append(j)
                    added = True
                    break
            if not added:
                n_steps.append(self.n_step)
        batch = [self.replay_memory[i] for i in indexes]
        states = torch.stack([torch.stack([self.replay_memory[(i - j - 1) % memory_size][0] for j in range(4)]) for i in indexes])
        next_states = torch.stack([torch.stack([self.replay_memory[(indexes[i] - j + n_steps[i] - 1) % memory_size][0] for j in range(4)]) for i in range(self.batch_size)])
        actions = torch.tensor([b[1] for b in batch], dtype=torch.long)
        rewards = torch.tensor([sum([self.replay_memory[(indexes[i] + j) % memory_size][2] * self.gamma ** j for j in range(n_steps[i])]) for i in range(self.batch_size)], dtype=torch.float32)
        done = torch.tensor([self.replay_memory[(indexes[i] + n_steps[i] - 1) % memory_size][3] for i in range(self.batch_size)], dtype=torch.bool)
        
        q_values = self.nn(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        self.optimizer.zero_grad()

        with torch.no_grad():
            next_actions = self.nn(next_states).argmax(dim=1)
            next_q_values = self.target_nn(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            discounted_final_values = torch.tensor([self.gamma ** n for n in n_steps])
            targets = rewards + discounted_final_values * next_q_values * ~done
        
        loss = torch.nn.functional.smooth_l1_loss(q_values, targets)
        with self.nn_lock:
            loss.backward()
            self.optimizer.step()

    def save(self):
        with self.best_lock:
            torch.save(self.nn.state_dict(), self.fileName)
            with open(os.path.splitext(self.fileName)[0] + ".txt", "w") as f:
                f.write(str(self.steps) + '\n')
                f.write(str(self.episode) + '\n')
                f.write(str(self.best_reward) + '\n')

    def load(self, fileName=None):
        if fileName is None:
            fileName = self.fileName
        self.nn.load_state_dict(torch.load(fileName))
        self.target_nn.load_state_dict(self.nn.state_dict())
        with open(os.path.splitext(fileName)[0] + ".txt") as f:
            self.steps = int(f.readline())
            self.episode = int(f.readline())
            self.best_reward = float(f.readline())

    def _test(self, episodes=10):
        with self.nn_lock:
            nn = copy.deepcopy(self.nn)
        nn.eval()
        while not self.interrupt:
            training_episode = self.episode
            with self.nn_lock:
                nn.load_state_dict(self.nn.state_dict())
            nn.eval()
            total_reward = 0
            for episode in range(episodes):
                obs, info = self.test_env.reset()
                obs = self.preprocessor(obs)
                sequence = [obs] * 4
                episode_reward = 0
                while not self.interrupt:
                    state = torch.stack(sequence)
                    with torch.no_grad():
                        q_values = nn(state.unsqueeze(0))
                    action = q_values.argmax(dim=1).item()
                    obs, reward, terminated, truncated, info = self.test_env.step(action)
                    episode_reward += reward
                    obs = self.preprocessor(obs)
                    sequence.pop()
                    sequence.insert(0, obs)
                    if terminated or truncated:
                        total_reward += episode_reward
                        break
                if self.interrupt:
                    episodes = episode 
                    break
            if episodes == 0:
                continue
            average = total_reward / episodes
            if self.reward_logger is not None:
                self.reward_logger(training_episode, average)
            with self.best_lock:
                if average > self.best_reward:
                    self.best_reward = average
                    torch.save(nn.state_dict(), 'best_' + self.fileName)

    def learn(self, timesteps, start=None):
        if start is not None:
            self.steps = start
        self.interrupt = False
        try:
            self.thread = Thread(target=self._test)
            self.thread.start()
            obs, info = self.env.reset()
            obs = self.preprocessor(obs)
            sequence = [obs] * 4
            while self.steps < timesteps:
                learning_rate = self.learning_rate(self.steps)
                epsilon = self.epsilon(self.steps)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = learning_rate
                current = torch.stack(sequence)
                action = self.select_action(current, epsilon)
                obs, reward, terminated, truncated, info = self.env.step(action)
                obs = self.preprocessor(obs)
                sequence.pop()
                sequence.insert(0, obs)
                
                if len(self.replay_memory) < self.memory_size:
                    self.replay_memory.append((sequence[0], action, reward, terminated, self.episode))
                else:
                    self.replay_memory[self.replay_index] = (sequence[0], action, reward, terminated, self.episode)
                self.replay_index = (self.replay_index + 1) % self.memory_size
                
                self.gradient_step()
                if self.steps % 1000 == 0:
                    self.target_nn.load_state_dict(self.nn.state_dict())
                    self.target_nn.eval()
                
                if terminated or truncated:
                    self.episode += 1
                    obs, info = self.env.reset()
                    obs = self.preprocessor(obs)
                    sequence = [obs] * 4
                self.steps += 1
                if self.steps % 1000 == 0:
                    self.save()
                if self.debug and self.steps % 10 == 0:
                    print(f"Steps: {self.steps}")
        except KeyboardInterrupt:
            pass
        finally:
            self.interrupt = True
            self.thread.join()
