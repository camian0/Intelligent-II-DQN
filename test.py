import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")


# Modelo de Red Neuronal
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)  # Aumentar neuronas
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))  # tanh para estabilizar activaciones
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# Clase DQNAgent
class DQNAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        replay_size=2000,
        batch_size=256,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.98,
        epsilon_min=0.01,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.replay_buffer = deque(maxlen=replay_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.loss_history = []

        self.q_network = DQN(state_dim, action_dim).to(device)
        self.target_network = DQN(state_dim, action_dim).to(device)
        self.update_target_network()
        self.optimizer = optim.Adam(
            self.q_network.parameters(), lr=0.0003
        )  # Reduzco lr
        self.criterion = nn.MSELoss()

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        state = torch.FloatTensor(state).to(device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_network(state)
        return torch.argmax(q_values).item()

    def store_experience(self, state, action, reward, next_state, terminated):
        self.replay_buffer.append((state, action, reward, next_state, terminated))

    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, terminateds = zip(*batch)

        states = torch.FloatTensor(np.array(states)).to(device)
        next_states = torch.FloatTensor(np.array(next_states)).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        actions = torch.LongTensor(actions).to(device)
        terminateds = torch.FloatTensor(terminateds).to(device)

        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = self.target_network(next_states).max(1)[0]
        targets = rewards + self.gamma * next_q_values * (1 - terminateds)

        loss = self.criterion(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.loss_history.append(loss.item())


def train_dqn(env_name="Acrobot-v1", episodes=500, target_update=10):
    env = gym.make(env_name, render_mode="rgb_array")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(state_dim, action_dim)
    scores = []

    for episode in range(episodes):
        state, _ = env.reset()
        episode_score = 0

        while True:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)

            # Recompensa personalizada
            target_height = -np.cos(next_state[0]) - np.cos(
                next_state[1] + next_state[0]
            )
            reward += target_height * 0.5  # Incentiva altura
            if terminated:
                reward = -10 if target_height < 1.0 else 10  # Penaliza o recompensa

            agent.store_experience(state, action, reward, next_state, terminated)
            agent.train()

            state = next_state
            episode_score += reward

            if terminated or truncated:
                break

        scores.append(episode_score)
        agent.epsilon = max(agent.epsilon * agent.epsilon_decay, agent.epsilon_min)

        if episode % target_update == 0:
            agent.update_target_network()

        print(
            f"Episode {episode}, Score: {episode_score:.2f}, Epsilon: {agent.epsilon:.2f}"
        )

    plt.plot(scores)
    plt.title("Evolución del Puntaje")
    plt.xlabel("Episodio")
    plt.ylabel("Puntaje")
    plt.grid()
    plt.show()


train_dqn(episodes=500)
