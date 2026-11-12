from cnn import CNN
import cv2
import gymnasium as gym
import torch
import random

torch.set_num_threads(4)
torch.set_num_interop_threads(1)

def preprocessor(x):
    return torch.from_numpy(cv2.cvtColor(x, cv2.COLOR_RGB2GRAY) / 255).float()

def test_dqn(env, fileName="model.pth"):
    model = CNN((4, 96, 96), env.action_space.n)
    model.load_state_dict(torch.load(fileName))
    model.eval()
    total_reward = 0
    episodes = 10
    for episode in range(episodes):
        obs, info = env.reset()
        obs = preprocessor(obs)
        sequence = [obs] * 4
        episode_reward = 0
        while True:
            state = torch.stack(sequence)
            with torch.no_grad():
                q_values = model(state.unsqueeze(0))
            action = q_values.argmax(dim=1).item()
            # max_q = q_values.squeeze()[action]
            # print(action, ',', max_q)
            # action = random.randint(0, env.action_space.n-1)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            obs = preprocessor(obs)
            sequence.pop()
            sequence.insert(0, obs)
            if terminated or truncated:
                print(f'Episode {episode + 1}: {episode_reward}')
                total_reward += episode_reward
                break
    print(f'Average reward: {total_reward / episodes}')

env = gym.make('CarRacing-v3', continuous=False)
test_dqn(env)
