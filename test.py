import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import gymnasium as gym
import random
from collections import deque
import matplotlib.pyplot as plt

# Crear el entorno
env = gym.make("Acrobot-v1")
num_actions = env.action_space.n
state_shape = env.observation_space.shape


# Definir la red neuronal para aproximar la función Q
def create_q_model():
    model = tf.keras.Sequential(
        [
            layers.Dense(
                128,
                activation="relu",
                kernel_initializer="he_uniform",
                input_shape=state_shape,
            ),
            layers.Dense(128, activation="relu", kernel_initializer="he_uniform"),
            layers.Dense(
                num_actions, activation="linear", kernel_initializer="he_uniform"
            ),
        ]
    )
    return model


# Parámetros de entrenamiento
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
batch_size = 64
max_steps_per_episode = 500
replay_buffer = deque(maxlen=10000)
learning_rate = 0.001


# Learning rate scheduler
def adjust_learning_rate(optimizer, episode, decay_rate=0.99):
    new_lr = learning_rate * (decay_rate ** (episode // 10))  # Decae cada 10 episodios
    optimizer.learning_rate = new_lr


# Crear los modelos de red Q
q_model = create_q_model()
target_q_model = create_q_model()
target_q_model.set_weights(q_model.get_weights())

# Definir el optimizador y la función de pérdida
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
loss_function = tf.keras.losses.MeanSquaredError()


# Función para seleccionar una acción
def select_action(state, epsilon):
    if np.random.rand() < epsilon:
        return np.random.choice(num_actions)
    q_values = q_model.predict(state[np.newaxis], verbose=0)
    return np.argmax(q_values[0])


# Función para entrenar el modelo
def train_step():
    if len(replay_buffer) < batch_size:
        return 0.0  # No hay suficiente experiencia

    batch = random.sample(replay_buffer, batch_size)
    states, actions, rewards, next_states, dones = map(np.array, zip(*batch))

    # Calcular los valores Q objetivo
    next_q_values = target_q_model.predict(next_states, verbose=0)
    max_next_q_values = np.max(next_q_values, axis=1)
    target_qs = rewards + (1 - dones) * gamma * max_next_q_values

    masks = tf.one_hot(actions, num_actions)

    with tf.GradientTape() as tape:
        q_values = q_model(states)
        q_action = tf.reduce_sum(q_values * masks, axis=1)
        loss = loss_function(target_qs, q_action)

    grads = tape.gradient(loss, q_model.trainable_variables)
    optimizer.apply_gradients(zip(grads, q_model.trainable_variables))

    return loss.numpy()


# Entrenamiento del agente
num_episodes = 5
target_update_freq = 10
scores = []
avg_rewards = []

for episode in range(num_episodes):
    state = env.reset()[0]
    total_reward = 0

    for step in range(max_steps_per_episode):
        action = select_action(state, epsilon)
        next_state, reward, done, _, _ = env.step(action)
        replay_buffer.append((state, action, reward, next_state, done))
        state = next_state
        total_reward += reward

        train_step()

        if done:
            break

    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    scores.append(total_reward)

    # Actualizar learning rate dinámicamente
    adjust_learning_rate(optimizer, episode)

    # Actualizar la red objetivo periódicamente
    if episode % target_update_freq == 0:
        target_q_model.set_weights(q_model.get_weights())

    # Monitorear recompensa promedio
    avg_reward = np.mean(scores[-20:]) if len(scores) >= 20 else np.mean(scores)
    avg_rewards.append(avg_reward)

    print(
        f"Episodio {episode + 1}: Recompensa = {total_reward:.2f}, Promedio (20) = {avg_reward:.2f}, Epsilon = {epsilon:.2f}, LR = {optimizer.learning_rate.numpy():.5f}"
    )

env.close()


# Graficar métricas
def plot_metrics(scores, avg_rewards):
    plt.figure(figsize=(12, 5))

    # Gráfico de puntajes
    plt.subplot(1, 2, 1)
    plt.plot(scores, label="Recompensa por episodio")
    plt.xlabel("Episodios")
    plt.ylabel("Recompensa")
    plt.legend()
    plt.grid()

    # Gráfico de promedio de recompensa
    plt.subplot(1, 2, 2)
    plt.plot(avg_rewards, label="Recompensa promedio (ventana de 20)")
    plt.xlabel("Episodios")
    plt.ylabel("Recompensa Promedio")
    plt.legend()
    plt.grid()

    plt.show()


plot_metrics(scores, avg_rewards)
