#!/usr/bin/env python3
"""Module that draws a red cubic curve"""
import numpy as np
import matplotlib.pyplot as plt


def line():
    """
    Draw a red cubic curve for x
    ranging from 0 to 10
    """
    y = np.arange(0, 11) ** 3

    plt.figure(figsize=(6.4, 4.8))
    plt.plot(y, color='red')
    plt.xlim(0, 10)
    plt.show()
