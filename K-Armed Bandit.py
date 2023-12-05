import numpy as np
import matplotlib.pyplot as plt

# Uniform Distribution Bandits
def simulate_bandits_uniform(K, N):
    bandits = []
    
    for i in range(K):
        low = 0  # Lower bound for the uniform distribution 
        high = 5  # Upper bound for the uniform distribution 
        bandit_rewards = np.random.uniform(low, high, size=N)
        bandits.append({'low': low, 'high': high, 'rewards': bandit_rewards})

    return bandits

# Normal Distribution Bandits
def simulate_bandits_normal(K, N):
    bandits = []
    
    # Simulate bandits
    for i in range(K):
        mean = np.random.uniform(0, 5)  # Adjust the range for mean as needed
        std_dev = np.random.uniform(0.5, 2)  # Adjust the range for standard deviation as needed
        bandit_rewards = np.random.normal(loc=mean, scale=std_dev, size=N)
        bandits.append({'mean': mean, 'std_dev': std_dev, 'rewards': bandit_rewards})

    return bandits

def explore_exploit_random(K, N):
    total_reward = 0
    lever_choices = []
    bandits = simulate_bandits_normal(K, N)

    for t in range(N):
        lever = np.random.randint(1, K+1)  # Randomly choose a lever
        reward = bandits[lever - 1]['rewards'][t]
        total_reward += reward
        lever_choices.append((lever, reward))

    return total_reward, lever_choices

def explore_exploit_pure_exploitation(K, N):
    total_reward = 0
    lever_choices = []
    bandits = simulate_bandits_normal(K, N)

    for t in range(N):
        if t < K:
            lever = t + 1  # In the first K turns, pull corresponding levers
        else:
            lever = np.argmax([np.mean(b['rewards'][:t]) for b in bandits]) + 1

        reward = bandits[lever - 1]['rewards'][t]
        total_reward += reward
        lever_choices.append((lever, reward))

    return total_reward, lever_choices

def explore_exploit_epsilon_greedy(K, N, epsilon):
    total_reward = 0
    lever_choices = []
    bandits = simulate_bandits_normal(K, N)

    for t in range(N):
        if np.random.rand() < epsilon:
            lever = np.random.randint(1, K+1)  # Exploration
        else:
            lever = np.argmax([np.mean(b['rewards'][:t]) for b in bandits]) + 1  # Exploitation

        reward = bandits[lever - 1]['rewards'][t]
        total_reward += reward
        lever_choices.append((lever, reward))

    return total_reward, lever_choices

def explore_exploit_decay_epsilon_greedy(K, N, initial_epsilon, decay_rate):
    total_reward = 0
    lever_choices = []
    bandits = simulate_bandits_normal(K, N)
    epsilon = initial_epsilon

    for t in range(N):
        if np.random.rand() < epsilon:
            lever = np.random.randint(1, K+1)  # Exploration
        else:
            lever = np.argmax([np.mean(b['rewards'][:t]) for b in bandits]) + 1  # Exploitation

        reward = bandits[lever - 1]['rewards'][t]
        total_reward += reward
        lever_choices.append((lever, reward))

        # Decay epsilon
        epsilon *= decay_rate

    return total_reward, lever_choices


def calculate_average_rewards(lever_choices, N):
    cumulative_rewards = np.cumsum([reward for _, reward in lever_choices])
    average_rewards = cumulative_rewards / np.arange(1, N + 1)
    return average_rewards

def plot_average_rewards(average_rewards, label):
    plt.plot(np.arange(1, len(average_rewards) + 1), average_rewards, label=label)

# Constants
K = 10  # Number of levers
N = 1000  # Number of pulls
epsilon = 0.5  # Epsilon value for epsilon-greedy strategy
decay_rate = 0.95  # Decay rate for decay epsilon-greedy strategy


# Scenario (a): Random
total_reward_random, lever_choices_random = explore_exploit_random(K, N)
average_rewards_random = calculate_average_rewards(lever_choices_random, N)
plot_average_rewards(average_rewards_random, 'Random Exploration')
print("Total reward (Random):", total_reward_random)

# Scenario (b): Pure Exploitation
total_reward_pure_exploitation, lever_choices_pure_exploitation = explore_exploit_pure_exploitation(K, N)
average_rewards_pure_exploitation = calculate_average_rewards(lever_choices_pure_exploitation, N)
plot_average_rewards(average_rewards_pure_exploitation, 'Pure Exploitation')
print("Total reward (Pure Exploitation):", total_reward_pure_exploitation)

# Scenario (c): Epsilon-Greedy
total_reward_epsilon_greedy, lever_choices_epsilon_greedy = explore_exploit_epsilon_greedy(K, N, epsilon)
average_rewards_epsilon_greedy = calculate_average_rewards(lever_choices_epsilon_greedy, N)
plot_average_rewards(average_rewards_epsilon_greedy, f'Epsilon-Greedy (epsilon={epsilon})')
print("Total reward (Epsilon-Greedy):", total_reward_epsilon_greedy)

# Scenario (d): Decay Epsilon-Greedy
total_reward_decay_epsilon_greedy, lever_choices_decay_epsilon_greedy = explore_exploit_decay_epsilon_greedy(K, N, epsilon, decay_rate)
average_rewards_decay_epsilon_greedy = calculate_average_rewards(lever_choices_decay_epsilon_greedy, N)
plot_average_rewards(average_rewards_decay_epsilon_greedy, f'Decay Epsilon-Greedy (initial_epsilon={epsilon}, decay_rate={decay_rate})')
print("Total reward (Decay Epsilon-Greedy):", total_reward_decay_epsilon_greedy)

# print(simulate_bandits(K, N))

# Plotting
plt.title('Average Reward vs. Pulls')
plt.xlabel('Number of Pulls')
plt.ylabel('Average Reward')
plt.legend()
plt.show()



