# 🌟 Q-learning 🌟

## 📝 Description 
Q-learning is a fundamental algorithm in reinforcement learning that aims to learn the value of an action in a particular state of an environment. The project focuses on the implementation of Q-learning for different environments, providing a practical approach to understanding the nuances of reinforcement learning.

The key functionalities include setting up the environment using OpenAI's Gym, initializing the Q-table, applying the epsilon-greedy policy for action selection, and training the agent through episodes to optimize its actions for maximum reward. This project is essential for anyone looking to delve into reinforcement learning, offering a hands-on experience with one of the most widely used algorithms in the field.

## 📚 Resources
- [An introduction to Reinforcement Learning](https://intranet.hbtn.io/rltoken/gOyUyOcWBr5QnuXhgJuVlw)
- [Simple Reinforcement Learning: Q-learning](https://intranet.hbtn.io/rltoken/5tkmJSVZIhEW4rlyfNK_Lw)
- [Markov Decision Processes (MDPs) - Structuring a Reinforcement Learning Problem](https://intranet.hbtn.io/rltoken/YUp-gcE1R9mC1cjjfvFh2w)
- [Expected Return - What Drives a Reinforcement Learning Agent in an MDP](https://intranet.hbtn.io/rltoken/BKs-9m9ep0sTz8V-EjoGrQ)
- [Policies and Value Functions - Good Actions for a Reinforcement Learning Agent](https://intranet.hbtn.io/rltoken/i3LsCe2sGI5kM1Qzr0d6Rg)
- [What do Reinforcement Learning Algorithms Learn - Optimal Policies](https://intranet.hbtn.io/rltoken/3nofOjrYYD1ghpMOEkG02g)
- [Q-Learning Explained - A Reinforcement Learning Technique](https://intranet.hbtn.io/rltoken/v3Bnyaow4gRx6K0os1GsSA)
- [Exploration vs. Exploitation - Learning the Optimal Reinforcement Learning Policy](https://intranet.hbtn.io/rltoken/igctuWgpbqhykidsjEwZxQ)
- [OpenAI Gym and Python for Q-learning - Reinforcement Learning Code Project](https://intranet.hbtn.io/rltoken/gaJ3jzi3XAz5sNSZAwLTjg)
- [Train Q-learning Agent with Python - Reinforcement Learning Code Project](https://intranet.hbtn.io/rltoken/K7JzzioxdqfLIl6z-SC5Rw)
- [Markov Decision Processes](https://intranet.hbtn.io/rltoken/IRn9ww1sX2MQSCsqwgUhSA)

## 🛠️ Technologies et Outils Utilisés
- **Python**: The primary programming language used for implementing Q-learning algorithms and interacting with the OpenAI Gym environments.
- **OpenAI Gym**: A toolkit for developing and comparing reinforcement learning algorithms. It provides the environments used for training the Q-learning agent.
- **NumPy**: A fundamental package for scientific computing with Python, used for handling the Q-table and other numerical computations.

## 📋 Prérequis
- ![Python](https://img.shields.io/badge/python-3.5_or_higher-blue)
- ![NumPy](https://img.shields.io/badge/numpy-1.15-blue)
- ![OpenAI Gym](https://img.shields.io/badge/gym-0.13-blue)

## 🚀 Installation et Configuration
1. **Clone the repository**:
    ```sh
    git clone https://github.com/CaroChoch/holbertonschool-machine_learning.git
    cd holbertonschool-machine_learning/reinforcement_learning/q_learning
    ```

2. **Install the required dependencies**:
    ```sh
    pip install --user gym
    pip install numpy
    ```

3. **Run the project**:
    ```sh
    python3 your_script.py
    ```

Ensure you have Python 3.5 or higher installed on your system. You can check your Python version with:
```sh
python3 --version
```

## 💡 Usage
- **Example usage 1** :
    ```sh
    python3 0-main.py
    ```
    This script loads the FrozenLake environment and displays the initial setup.

- **Example usage 2** :
    ```sh
    python3 1-main.py
    ```
    This script initializes the Q-table and shows its shape for different environment configurations.

- **Example usage 3** :
    ```sh
    python3 2-main.py
    ```
    This script demonstrates the epsilon-greedy policy for action selection in the Q-learning process.

## ✨ Main Features
1. **Load the Environment**:
    The project includes a function to load various configurations of the FrozenLake environment from OpenAI's Gym, with options for custom maps and slippery conditions.

2. **Initialize Q-table**:
    A function to initialize the Q-table as a numpy array filled with zeros, prepared for the Q-learning algorithm.

3. **Epsilon Greedy**:
    Implements the epsilon-greedy policy for balancing exploration and exploitation in action selection.

4. **Q-learning Training**:
    A comprehensive function to train the agent using the Q-learning algorithm, updating the Q-table based on the agent's experiences.

5. **Play**:
    Allows the trained agent to play an episode in the environment, displaying the state transitions and total rewards.

## 📝  Task List
| Number | Task | Description |
| ------ | ---------------------- | ------------------------------------------------------------------------------- |
| 0 | **Load the Environment** | Write a function `def load_frozen_lake(desc=None, map_name=None, is_slippery=False):` that loads the pre-made FrozenLakeEnv environment from OpenAI’s gym: desc is either None or a list of lists containing a custom description of the map to load for the environment; map_name is either None or a string containing the pre-made map to load; if both desc and map_name are None, the environment will load a randomly generated 8x8 map; is_slippery is a boolean to determine if the ice is slippery; Returns: the environment |
| 1 | **Initialize Q-table** | Write a function `def q_init(env):` that initializes the Q-table: env is the FrozenLakeEnv instance; Returns: the Q-table as a numpy.ndarray of zeros |
| 2 | **Epsilon Greedy** | Write a function `def epsilon_greedy(Q, state, epsilon):` that uses epsilon-greedy to determine the next action: Q is a numpy.ndarray containing the q-table; state is the current state; epsilon is the epsilon to use for the calculation; You should sample p with numpy.random.uniform to determine if your algorithm should explore or exploit; If exploring, you should pick the next action with numpy.random.randint from all possible actions; Returns: the next action index |
| 3 | **Q-learning** | Write the function `def train(env, Q, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):` that performs Q-learning: env is the FrozenLakeEnv instance; Q is a numpy.ndarray containing the Q-table; episodes is the total number of episodes to train over; max_steps is the maximum number of steps per episode; alpha is the learning rate; gamma is the discount rate; epsilon is the initial threshold for epsilon greedy; min_epsilon is the minimum value that epsilon should decay to; epsilon_decay is the decay rate for updating epsilon between episodes; When the agent falls in a hole, the reward should be updated to be -1; Returns: Q, total_rewards; Q is the updated Q-table; total_rewards is a list containing the rewards per episode |
| 4 | **Play** | Write a function `def play(env, Q, max_steps=100):` that has the trained agent play an episode: env is the FrozenLakeEnv instance; Q is a numpy.ndarray containing the Q-table; max_steps is the maximum number of steps in the episode; Each state of the board should be displayed via the console; You should always exploit the Q-table; Returns: the total rewards for the episode |

## 📬 Contact
- LinkedIn Profile: [Caroline CHOCHOY](https://www.linkedin.com/in/caroline-chochoy62/)
