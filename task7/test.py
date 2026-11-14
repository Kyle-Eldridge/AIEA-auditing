from cnn import CNN
import cv2
import gymnasium as gym
import torch
import random
from threading import Thread

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

def preprocessor(x):
    return torch.from_numpy(cv2.cvtColor(x, cv2.COLOR_RGB2GRAY) / 255).float()

def run_episode(env, model, output_list, index):
    frame_skip = 2
    obs, info = env.reset()
    obs = preprocessor(obs)
    sequence = [obs] * 4
    episode_reward = 0
    while True:
        state = torch.stack(sequence)
        with torch.no_grad():
            actions, _ = model(state.unsqueeze(0))
        action = actions.argmax(dim=1).item()
        reward = 0
        for _ in range(frame_skip):
            obs, step_reward, terminated, truncated, info = env.step(action)
            reward += step_reward
            if terminated or truncated:
                break
        episode_reward += reward
        obs = preprocessor(obs)
        sequence.pop(0)
        sequence.append(obs)
        if terminated or truncated:
            output_list[index] = episode_reward
            break

def test(env_supplier, file="model.pth"):
    model = CNN((4, 96, 96), env_supplier().action_space.n)
    model.load_state_dict(torch.load(file))
    model.eval()
    episodes = 10
    rewards = [0] * episodes
    threads = [Thread(target=run_episode, args=(env_supplier(), model, rewards, i)) for i in range(episodes)]
    for t in threads:
        t.start()
    for i in range(len(threads)):
        threads[i].join()
        print(f"Episode {i + 1}: {rewards[i]}")
    print(f'Average reward: {sum(rewards) / episodes}')

env_supplier = lambda: gym.make('CarRacing-v3', continuous=False)
test(env_supplier)
