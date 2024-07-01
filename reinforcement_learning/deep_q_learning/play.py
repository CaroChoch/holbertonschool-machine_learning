#!/usr/bin/env python3
"""
Playing Atari Breakout using Deep Q-Learning with Keras-RL2
"""
import gym
import tensorflow as tf
import tensorflow.keras as K

# Import build_model and build_agent functions from the train module
build_model = __import__('train').build_model
build_agent = __import__('train').build_agent

# Constants for input shape and window length
INPUT_SHAPE = (84, 84)
WINDOW_LENGTH = 4


def play_breakout(env_name='Breakout-v4', model_weights='policy.h5',
                  episodes=10, render_mode='human'):
    """
    Function to play Atari Breakout using a pre-trained model.

    Parameters:
        env_name (str): Name of the gym environment to use.
        model_weights (str): Path to the pre-trained model weights.
        episodes (int): Number of episodes to play.
        render_mode (str): Mode to render the environment.
    """
    # Create the gym environment
    env = gym.make(env_name, render_mode=render_mode)

    # Reset the environment and get the initial observation and info
    observation, info = env.reset(return_info=True)

    # Get the number of possible actions in the environment
    actions = env.action_space.n

    # Build the model using the number of actions
    model = build_model(actions)

    # Build the DQN agent with the model and actions
    dqn = build_agent(model, actions)

    # Compile the DQN agent with the legacy Adam optimizer
    # and mean absolute error metric
    dqn.compile(tf.keras.optimizers.legacy.Adam(
        learning_rate=1e-4),
        metrics=['mae'])

    # Load the pre-trained weights into the DQN agent
    dqn.load_weights(model_weights)

    # Test the DQN agent in the environment for a specified number of episodes
    dqn.test(env, nb_episodes=episodes, visualize=False)


if __name__ == '__main__':
    # Call the play_breakout function to start playing
    play_breakout()
