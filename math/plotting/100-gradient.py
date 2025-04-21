#!/usr/bin/env python3
"""Displays a map of altitude dispersion on a mountain"""
import numpy as np
import matplotlib.pyplot as plt


def gradient():
    """
    Displays a map of altitude dispersion on a mountain
    """
    np.random.seed(5)

    x = np.random.randn(2000) * 10
    y = np.random.randn(2000) * 10
    z = np.random.rand(2000) + 40 - np.sqrt(np.square(x) + np.square(y))

    plt.xlabel('x coordinate (m)')
    plt.ylabel('y coordinate (m)')
    plt.title('Mountain Elevation')
    plt.colorbar(plt.scatter(x, y, c=z), label="elevation (m)")
    plt.show()
