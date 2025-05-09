#!/usr/bin/env python3
""" plays an episode of FrozenLake using Q-learning """
import numpy as np


def play(env, Q, max_steps=100):
    """
    Plays an episode with a trained agent.

    Arguments:
        - env: FrozenLakeEnv instance (avec render_mode="ansi")
        - Q: numpy.ndarray contenant la Q-table
        - max_steps: nombre maximal de pas à jouer

    Returns:
        total_rewards: float, somme des récompenses obtenues
        rendered_outputs: list de str, chaque str est soit tout le plateau
                          ASCII (avec saut de ligne initial), soit la ligne
                          d’action (e.g. "  (Down)")
    """
    # Réinit de l'env et récupération de l'état
    state = env.reset()[0]
    total_rewards = 0
    actions = ["Left", "Down", "Right", "Up"]
    rendered_outputs = []

    for _ in range(max_steps):
        # 1) on capture et stocke le plateau complet
        rendered_outputs.append(env.render())

        # 2) on choisit et stocke l'action
        action = int(np.argmax(Q[state]))
        rendered_outputs.append(f"  ({actions[action]})")

        # 3) on exécute la transition
        next_state, reward, terminated, truncated, _ = env.step(action)
        total_rewards += reward
        state = next_state

        # 4) si fin d’épisode, on affiche le plateau final puis break
        if terminated or truncated:
            rendered_outputs.append(env.render())
            break

    # 5) on ferme l'environnement
    env.close()
    return total_rewards, rendered_outputs
