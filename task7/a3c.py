import copy
from threading import Thread, Lock
import torch
import torch.optim as optim
from torch.nn.functional import smooth_l1_loss
import gymnasium as gym
import numpy as np
import random

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

class A3CThread(Thread):
    def __init__(self, env, a3c):
        super().__init__()
        self.env = env
        self.a3c = a3c
    
    def select_action(self, obs):
        with torch.no_grad():
            actions, _ = self.nn(obs.unsqueeze(0))
            probabilities = torch.softmax(actions, dim=1)
            return torch.multinomial(probabilities, 1).item()

    def get_value(self, obs):
        with torch.no_grad():
            _, value = self.nn(obs.unsqueeze(0))
            return value.item()

    def accumulate_gradients(self, state, action, reward, updates):
        actions, value = self.nn(state.unsqueeze(0))
        advantage = reward - value.detach()
        log_probs = torch.log_softmax(actions, dim=1)
        log_prob = log_probs[0, action]
        probs = torch.softmax(actions, dim=-1)

        policy_loss = -log_prob * advantage
        value_loss = smooth_l1_loss(value, value.new_tensor([reward]))
        entropy = -(probs * log_probs).sum(dim=-1).mean()
        loss = (policy_loss + self.a3c.value_loss_coef * value_loss - self.a3c.entropy_coef * entropy) / updates
        loss.backward()

    def run(self):
        self.nn = copy.deepcopy(self.a3c.nn)
        steps = 0
        total_reward = 0
        obs, info = self.env.reset()
        obs = self.a3c.preprocessor(obs)
        state = [obs] * 4

        while not self.a3c.interrupted and self.a3c.total_steps < self.a3c.timesteps:
            with self.a3c.nn_lock:
                self.nn.load_state_dict(self.a3c.nn.state_dict())
            self.nn.zero_grad()
            start = steps
            terminated = False
            truncated = False
            states = [torch.stack(state)]
            rewards = []
            actions = []

            while not terminated and not truncated and steps - start < self.a3c.t_max:
                action = self.select_action(states[-1])
                actions.append(action)
                reward = 0
                for _ in range(self.a3c.frame_skip):
                    obs, step_reward, terminated, truncated, info = self.env.step(action)
                    reward += step_reward
                    if terminated or truncated:
                        break
                obs = self.a3c.preprocessor(obs)
                state.pop(0)
                state.append(obs)
                states.append(torch.stack(state))
                rewards.append(reward)
                total_reward += reward
                steps += 1

            save = False
            with self.a3c.step_lock:
                self.a3c.total_steps += steps - start
                save = self.a3c.total_steps % self.a3c.save_frequency < steps - start
            if self.a3c.debug:
                print(f"Steps: {steps} local, {self.a3c.total_steps} global")
            
            updates = len(rewards)
            R = 0 if terminated else self.get_value(states[-1])
            for i in range(updates - 1, -1, -1):
                R = rewards[i] + self.a3c.gamma * R
                self.accumulate_gradients(states[i], actions[i], R, updates)
            
            with self.a3c.nn_lock:
                self.a3c.optimizer.zero_grad()
                for a3c_parameter, local_parameter in zip(self.a3c.nn.parameters(), self.nn.parameters()):
                    a3c_parameter.grad = local_parameter.grad
                self.a3c.optimizer.step()
                if save:
                    torch.save(self.a3c.nn.state_dict(), self.a3c.file)
            
            if terminated or truncated:
                self.a3c.reward_logger(total_reward)
                total_reward = 0
                obs, info = self.env.reset()
                obs = self.a3c.preprocessor(obs)
                state = [obs] * 4


class A3C:
    def __init__(self, env_supplier, nn, num_threads=8, learning_rate=1e-4, gamma=0.98, t_max=5, value_loss_coef=0.5, entropy_coef=0.01, frame_skip=2, file="model.pth", save_frequency=1000, preprocessor=None, reward_logger=None, debug=False):
        self.env_supplier = env_supplier
        self.nn = nn
        self.num_threads = num_threads
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.t_max = t_max
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.frame_skip = frame_skip
        self.file = file
        self.save_frequency = save_frequency
        self.preprocessor = preprocessor if preprocessor else lambda x: x
        self.reward_logger = reward_logger if reward_logger else lambda reward: None
        self.debug = debug
        self.optimizer = optim.Adam(self.nn.parameters(), lr=self.learning_rate)
        self.nn_lock = Lock()
        self.step_lock = Lock()
    
    def learn(self, timesteps, start=0):
        self.total_steps = start
        self.threads = [A3CThread(self.env_supplier(), self) for _ in range(self.num_threads)]
        self.timesteps = timesteps
        self.interrupted = False
        for thread in self.threads:
            thread.start()
        while True:
            try:
                for thread in self.threads:
                    thread.join()
                break
            except KeyboardInterrupt:
                self.interrupted = True
